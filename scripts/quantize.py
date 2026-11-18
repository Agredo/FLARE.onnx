"""Produce fp16, int8, and int4 variants of the published float32 model."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import traceback
from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "flare.onnx"
OUTPUT_DIR = ROOT / "models"
MANIFEST = OUTPUT_DIR / "manifest.json"

VARIANT_FILES = {
    "fp16": "flare.fp16.onnx",
    "int8": "flare.int8.onnx",
    "int4": "flare.int4.onnx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_model_proto(candidate):
    """Unwrap the various return shapes used by the ONNX Runtime quantizers."""
    if candidate is None:
        return None
    inner = getattr(candidate, "model", None)
    if inner is not None and hasattr(inner, "graph"):
        return inner
    if hasattr(candidate, "graph"):
        return candidate
    return None


def _convert_fp16(module, function_name: str, model, label: str):
    function = getattr(module, function_name)
    supported = inspect.signature(function).parameters
    kwargs = {key: value for key, value in {
        "keep_io_types": True,
        "disable_shape_infer": False,
        "use_symbolic_shape_infer": False,
    }.items() if key in supported}
    return function(model, **kwargs), label


def _toposort(model) -> int:
    """Order the nodes so that every input is produced before it is used.

    The float16 converter inserts cast nodes without reordering the graph, which the ONNX checker
    rejects even though ONNX Runtime tolerates it. Returns the number of nodes that had to move.
    """
    graph = model.graph
    available = {initializer.name for initializer in graph.initializer}
    available.update(input_.name for input_ in graph.input)

    ordered = []
    pending = list(graph.node)
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            ordered.extend(pending)
            break
        for node in ready:
            available.update(node.output)
        ordered.extend(ready)
        blocked = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in blocked]

    original = [node.name for node in graph.node]
    if [node.name for node in ordered] == original:
        return 0

    del graph.node[:]
    graph.node.extend(ordered)
    moved = sum(1 for before, after in zip(original, [node.name for node in ordered]) if before != after)
    return max(moved, 1)


def quantize_fp16(destination: Path, _block_size: int) -> str:
    model = onnx.load(str(SOURCE))

    try:
        from onnxruntime.transformers import float16 as ort_float16

        converted, recipe = _convert_fp16(
            ort_float16, "convert_float_to_float16", model, "onnxruntime convert_float_to_float16"
        )
    except Exception as error:  # noqa: BLE001 - fall back to the reference converter
        print(f"onnxruntime fp16 converter unavailable ({type(error).__name__}: {error})")
        from onnxconverter_common import float16 as reference_float16

        converted, recipe = _convert_fp16(
            reference_float16,
            "convert_float_to_float16",
            model,
            "onnxconverter_common convert_float_to_float16",
        )

    _toposort(converted)
    onnx.save(converted, str(destination))
    return f"{recipe}(keep_io_types=True)"


def quantize_int8(destination: Path, _block_size: int) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(SOURCE), str(destination), weight_type=QuantType.QInt8)
    return "quantize_dynamic(weight_type=QInt8)"


WEIGHT_ONLY_MODULES = (
    "onnxruntime.quantization.matmul_nbits_quantizer",
    "onnxruntime.quantization.matmul_4bits_quantizer",
)
WEIGHT_ONLY_CLASSES = ("MatMulNBitsQuantizer", "MatMul4BitsQuantizer")


def _load_weight_only_quantizer():
    """Locate the MatMul bits-quantizer; ONNX Runtime renamed it across releases."""
    import importlib

    attempts = []
    for module_name in WEIGHT_ONLY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - report every candidate
            attempts.append(f"{module_name} ({type(error).__name__})")
            continue
        for class_name in WEIGHT_ONLY_CLASSES:
            candidate = getattr(module, class_name, None)
            if inspect.isclass(candidate):
                return candidate, f"{module_name}.{class_name}"
    raise ImportError("no MatMul weight-only quantizer found in onnxruntime; tried " + ", ".join(attempts))


def _rewrite_gemm_to_matmul(model) -> int:
    """Express alpha=1/beta=1 Gemm nodes as MatMul, so weight-only quantization can see them.

    The quantizer only targets MatMul, while PyTorch exports most linear layers as Gemm. The weight
    initializer is transposed here instead of inserting a Transpose node, which keeps the
    MatMul-with-constant-RHS pattern intact.
    """
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    replacements = []
    added = []
    rewritten = 0

    for node in graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2:
            replacements.append(node)
            continue
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        weight = initializers.get(node.input[1])
        if (
            weight is None
            or attributes.get("transA", 0)
            or attributes.get("alpha", 1.0) != 1.0
            or attributes.get("beta", 1.0) != 1.0
        ):
            replacements.append(node)
            continue
        array = numpy_helper.to_array(weight)
        if array.ndim != 2:
            replacements.append(node)
            continue
        if attributes.get("transB", 0):
            array = array.T

        name = f"{node.input[1]}_matmul{rewritten}"
        added.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        if bias is None:
            replacements.append(helper.make_node("MatMul", [node.input[0], name], [node.output[0]], name=node.name))
        else:
            intermediate = f"{node.output[0]}_matmul"
            replacements.append(
                helper.make_node("MatMul", [node.input[0], name], [intermediate], name=f"{node.name}_matmul")
            )
            replacements.append(helper.make_node("Add", [intermediate, bias], [node.output[0]], name=node.name))
        rewritten += 1

    if rewritten:
        del graph.node[:]
        graph.node.extend(replacements)
        graph.initializer.extend(added)
    return rewritten


def quantize_int4(destination: Path, block_size: int) -> str:
    quantizer_class, fully_qualified = _load_weight_only_quantizer()

    model = onnx.load(str(SOURCE))
    rewritten = _rewrite_gemm_to_matmul(model)
    print(f"  rewrote {rewritten} Gemm node(s) to MatMul for weight-only quantization")
    supported = inspect.signature(quantizer_class.__init__).parameters
    kwargs = {"bits": 4, "block_size": block_size, "is_symmetric": True, "nodes_to_exclude": []}
    quantizer = quantizer_class(model, **{k: v for k, v in kwargs.items() if k in supported})

    if hasattr(quantizer, "process"):
        quantizer.process()
    elif hasattr(quantizer, "quantize"):
        quantizer.quantize()
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"{fully_qualified} exposes neither process() nor quantize()")

    proto = None
    for attribute in ("model", "quantized_model"):
        proto = _to_model_proto(getattr(quantizer, attribute, None))
        if proto is not None:
            break
    if proto is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not obtain a quantized ModelProto")

    onnx.save(proto, str(destination))
    return (
        f"{fully_qualified.rsplit('.', 1)[-1]}(bits=4, block_size={block_size}, is_symmetric=True), "
        f"after rewriting {rewritten} Gemm node(s) to MatMul"
    )


BUILDERS = {"fp16": quantize_fp16, "int8": quantize_int8, "int4": quantize_int4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="fp16,int8,int4")
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"error: {SOURCE} is missing (check out the repository with Git LFS)", file=sys.stderr)
        return 1

    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in requested if name not in BUILDERS]
    if unknown:
        print(f"error: unknown variant(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    print(f"source: {SOURCE} ({SOURCE.stat().st_size} bytes)")
    print(f"python: {platform.python_version()}, onnx: {onnx.__version__}")

    for name in requested:
        destination = OUTPUT_DIR / VARIANT_FILES[name]
        if destination.exists():
            destination.unlink()
        print(f"\n== {name} -> {destination.name}")
        try:
            recipe = BUILDERS[name](destination, args.block_size)
            onnx.checker.check_model(onnx.load(str(destination)))
            size = destination.stat().st_size
            entries.append(
                {
                    "variant": name,
                    "file": destination.name,
                    "status": "built",
                    "recipe": recipe,
                    "bytes": size,
                    "sha256": sha256(destination),
                }
            )
            print(f"built {destination.name}: {size} bytes ({size / 1e6:.2f} MB)")
        except Exception as error:  # noqa: BLE001 - report and continue with the others
            traceback.print_exc()
            if destination.exists():
                destination.unlink()
            entries.append(
                {
                    "variant": name,
                    "file": None,
                    "status": "skipped",
                    "recipe": None,
                    "bytes": None,
                    "sha256": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"skipped {name}: {type(error).__name__}: {error}")

    manifest = {
        "source": {"file": SOURCE.name, "bytes": SOURCE.stat().st_size, "sha256": sha256(SOURCE)},
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    built = [entry for entry in entries if entry["status"] == "built"]
    print(f"\nbuilt {len(built)} of {len(requested)} variant(s); manifest: {MANIFEST}")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
