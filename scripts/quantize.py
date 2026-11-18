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

    onnx.save(converted, str(destination))
    return f"{recipe}(keep_io_types=True)"


def quantize_int8(destination: Path, _block_size: int) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(SOURCE), str(destination), weight_type=QuantType.QInt8)
    return "quantize_dynamic(weight_type=QInt8)"


def quantize_int4(destination: Path, block_size: int) -> str:
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

    model = onnx.load(str(SOURCE))
    supported = inspect.signature(MatMul4BitsQuantizer.__init__).parameters
    kwargs = {"block_size": block_size, "is_symmetric": True}
    if "nodes_to_exclude" in supported:
        kwargs["nodes_to_exclude"] = []
    quantizer = MatMul4BitsQuantizer(model, **{k: v for k, v in kwargs.items() if k in supported})

    if hasattr(quantizer, "process"):
        quantizer.process()
    elif hasattr(quantizer, "quantize"):
        quantizer.quantize()
    else:  # pragma: no cover - defensive
        raise RuntimeError("MatMul4BitsQuantizer exposes neither process() nor quantize()")

    proto = None
    for attribute in ("model", "quantized_model"):
        proto = _to_model_proto(getattr(quantizer, attribute, None))
        if proto is not None:
            break
    if proto is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not obtain a quantized ModelProto")

    onnx.save(proto, str(destination))
    return f"MatMul4BitsQuantizer(block_size={block_size}, is_symmetric=True)"


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
