"""Export FLARE to a single ONNX model and verify it against the original pipeline.

    python scripts/export_onnx.py [--ckpt pretrained_models/flare.ckpt] [--out flare.onnx]

The checkpoint is not part of this repository; download `pretrained_models/flare.ckpt`
from https://huggingface.co/spaces/HassounLab/FLARE/tree/main and keep it next to
`flare.onnx`, or pass a path with --ckpt.

Inputs  : spec [P,15] float32, atoms [B,N,78] float32, adj [B,N,N] float32, atom_mask [B,N] float32
Outputs : scores [B] float32, similarity [B,P,N] float32

Verification (if reference/reference.json exists, see make_reference.py):
pure-torch model and ONNX Runtime scores are compared with the scores of the original
DGL/dgllife implementation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from model import ATOM_IN_DIM, SPEC_IN_DIM, load_flare_onnx  # noqa: E402


def batch_molecules(molecules: list[dict]):
    b = len(molecules)
    n = max(m["num_atoms"] for m in molecules)
    atoms = np.zeros((b, n, ATOM_IN_DIM), dtype=np.float32)
    adj = np.zeros((b, n, n), dtype=np.float32)
    mask = np.zeros((b, n), dtype=np.float32)
    for i, m in enumerate(molecules):
        k = m["num_atoms"]
        atoms[i, :k] = np.asarray(m["atom_features"], dtype=np.float32)
        mask[i, :k] = 1.0
        for s, d in m["edges"]:
            adj[i, d, s] += 1.0  # message s -> d, summed at d (dgl GraphConv, norm='none')
    return atoms, adj, mask


def verify(model, onnx_path: Path, ref_path: Path, tol: float) -> None:
    import onnxruntime as ort

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    atoms, adj, mask = batch_molecules(ref["molecules"])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_torch = max_ort = 0.0
    top1 = 0
    for s in ref["spectra"]:
        spec = np.asarray(s["spec_features"], dtype=np.float32)
        expected = np.asarray(s["scores"], dtype=np.float32)
        with torch.no_grad():
            t_scores, _ = model(*(torch.from_numpy(a) for a in (spec, atoms, adj, mask)))
        o_scores, _ = sess.run(None, {"spec": spec, "atoms": atoms, "adj": adj, "atom_mask": mask})
        max_torch = max(max_torch, float(np.abs(t_scores.numpy() - expected).max()))
        max_ort = max(max_ort, float(np.abs(o_scores - expected).max()))
        true_idx = ref["candidates"].index(s["input"]["true_smiles"])
        top1 += int(np.argmax(o_scores) == true_idx)
    print(f"max |torch - original| = {max_torch:.2e}")
    print(f"max |onnx  - original| = {max_ort:.2e}")
    print(f"top-1 (onnx) {top1}/{len(ref['spectra'])}")
    if max_torch > tol or max_ort > tol:
        raise SystemExit(f"verification FAILED (tol {tol})")
    print("verification OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "pretrained_models" / "flare.ckpt"))
    ap.add_argument("--out", default=str(ROOT / "flare.onnx"))
    ap.add_argument("--reference", default=str(ROOT / "reference" / "reference.json"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    model, _ = load_flare_onnx(args.ckpt)
    dummy = (
        torch.rand(7, SPEC_IN_DIM),
        torch.rand(2, 5, ATOM_IN_DIM),
        torch.eye(5).repeat(2, 1, 1),
        torch.ones(2, 5),
    )
    kwargs = dict(
        input_names=["spec", "atoms", "adj", "atom_mask"],
        output_names=["scores", "similarity"],
        dynamic_axes={
            "spec": {0: "peaks"},
            "atoms": {0: "batch", 1: "atoms"},
            "adj": {0: "batch", 1: "atoms", 2: "atoms"},
            "atom_mask": {0: "batch", 1: "atoms"},
            "scores": {0: "batch"},
            "similarity": {0: "batch", 1: "peaks", 2: "atoms"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # torch >= 2.5: keep the TorchScript exporter
    torch.onnx.export(model, dummy, args.out, **kwargs)
    import onnx

    onnx.checker.check_model(onnx.load(args.out))
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    ref = Path(args.reference)
    if ref.exists():
        verify(model, Path(args.out), ref, args.tol)
    else:
        print(f"no reference at {ref}; run make_reference.py to verify")


if __name__ == "__main__":
    main()
