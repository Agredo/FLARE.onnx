"""Verify quantized variants against the float32 model and the golden fixture.

Every model is run over the 40 reference spectra against the 40 reference
candidates. Two deviations are reported per variant: against the published
float32 model (numerical drift caused by quantization) and against the original
PyTorch scores (parity with the research implementation).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tensors import candidate_tensors, load_fixture, reference_scores, spectrum_tensors

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "flare.onnx"
OUTPUT_DIR = ROOT / "models"
MANIFEST = OUTPUT_DIR / "manifest.json"

EXPECTED_INPUTS = ["spec", "atoms", "adj", "atom_mask"]

# Gates applied per variant: (max drift vs float32, minimum top-1 agreement).
DEFAULT_GATES = {
    "fp16": (5e-3, 40),
    "int8": (8e-2, 39),
    "int4": (2e-1, 38),
}

# The same tolerance the FLARE.Net test suite uses for the float32 reference model.
REFERENCE_TOLERANCE = 1e-4


def build_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def check_contract(session: ort.InferenceSession, name: str) -> None:
    inputs = [item.name for item in session.get_inputs()]
    missing = [item for item in EXPECTED_INPUTS if item not in inputs]
    if missing:
        raise RuntimeError(f"{name}: missing input(s) {missing}; found {inputs}")


def run_all(session, spec_list, atoms, adj, mask) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    rows = []
    for spec in spec_list:
        feed = {"spec": spec, "atoms": atoms, "adj": adj, "atom_mask": mask}
        scores = session.run(["scores"], feed)[0]
        rows.append(np.asarray(scores, dtype=np.float32))
    elapsed = (time.perf_counter() - started) * 1000.0
    return np.vstack(rows), elapsed


def main() -> int:
    if not MANIFEST.exists():
        print(f"error: {MANIFEST} is missing; run quantize.py first", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixture = load_fixture()
    spec_list = spectrum_tensors(fixture)
    atoms, adj, mask = candidate_tensors(fixture)
    expected = reference_scores(fixture)

    print(f"spectra: {len(spec_list)}, candidates: {atoms.shape[0]}, max atoms: {atoms.shape[1]}")
    print(f"onnxruntime: {ort.__version__}, providers: {[p for p in ort.get_available_providers()]}")

    models = [("float32", SOURCE, None)] + [
        (entry["variant"], OUTPUT_DIR / entry["file"], entry["variant"])
        for entry in manifest["entries"]
        if entry["status"] == "built"
    ]

    results = []
    baseline = None
    failure = False

    for name, path, variant in models:
        session = build_session(path)
        check_contract(session, name)
        scores, elapsed = run_all(session, spec_list, atoms, adj, mask)

        drift_vs_reference = float(np.max(np.abs(scores - expected)))
        top1_reference = int(np.sum(scores.argmax(axis=1) == expected.argmax(axis=1)))

        if baseline is None:
            baseline = scores
            drift_vs_fp32 = 0.0
            top1_fp32 = len(spec_list)
        else:
            drift_vs_fp32 = float(np.max(np.abs(scores - baseline)))
            top1_fp32 = int(np.sum(scores.argmax(axis=1) == baseline.argmax(axis=1)))

        size = path.stat().st_size
        gates = DEFAULT_GATES.get(variant or "")
        status = "baseline" if variant is None else "ok"
        if variant is None and drift_vs_reference > REFERENCE_TOLERANCE:
            status = f"FAILED (reference drift > {REFERENCE_TOLERANCE:g})"
            failure = True
        if gates is not None:
            limit, min_top1 = gates
            if drift_vs_fp32 > limit or top1_fp32 < min_top1:
                status = "FAILED"
                failure = True

        results.append(
            {
                "variant": name,
                "file": path.name,
                "bytes": size,
                "max_abs_vs_fp32": drift_vs_fp32,
                "max_abs_vs_reference": drift_vs_reference,
                "top1_vs_fp32": top1_fp32,
                "top1_vs_reference": top1_reference,
                "elapsed_ms": elapsed,
                "status": status,
            }
        )
        print(
            f"{name:8s} size={size:>10d} drift_fp32={drift_vs_fp32:.3e} "
            f"drift_ref={drift_vs_reference:.3e} top1={top1_fp32:2d}/{len(spec_list)} "
            f"time={elapsed:7.1f} ms [{status}]"
        )

    lines = [
        "## Quantized model verification",
        "",
        f"{len(spec_list)} spectra x {atoms.shape[0]} candidates, CPUExecutionProvider, "
        f"onnxruntime {ort.__version__}.",
        "",
        "| Variant | Size | vs float32 (max abs) | vs original (max abs) | Top-1 vs float32 | Top-1 vs original | 40 spectra | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['variant']} | {row['bytes'] / 1e6:.2f} MB | {row['max_abs_vs_fp32']:.3e} | "
            f"{row['max_abs_vs_reference']:.3e} | {row['top1_vs_fp32']}/{len(spec_list)} | "
            f"{row['top1_vs_reference']}/{len(spec_list)} | {row['elapsed_ms']:.0f} ms | {row['status']} |"
        )

    skipped = [entry for entry in manifest["entries"] if entry["status"] != "built"]
    if skipped:
        lines.append("")
        for entry in skipped:
            lines.append(f"- `{entry['variant']}` was not built: {entry.get('error', 'unknown reason')}")

    report = "\n".join(lines)
    print("\n" + report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
    )

    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
