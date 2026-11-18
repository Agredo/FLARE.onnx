"""Generate golden reference data with the ORIGINAL FLARE pipeline (DGL + dgllife).

The output JSON is used to validate
  * the ONNX export (export_onnx.py), and
  * the native C# port (Agredo/FLARE.Net tests).

Pipeline per spectrum (identical to the MassSpecGym/MIST preprocessing used for training):
  process_spec_file -> assign_subforms (20 ppm) -> add/replace precursor peak (intensity 1.1)
  -> sort by m/z -> SpecFormulaFeaturizer
Each spectrum is scored against all candidate SMILES with filip_similarity_batch.

This script imports the original FLARE code, so it needs a checkout of the research
project (the directory containing the `flare/` package) and the Python environment
listed in requirements-reference.txt. Point FLARE_SRC at that checkout:

    $env:FLARE_SRC = "C:\\src\\FLARE"
    python scripts/make_reference.py --n 40

The pre-built result for 40 spectra is already included in the reference data used by
scripts/verify.py (see scripts/fetch_reference.py), so running this script is only
necessary to regenerate or extend it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def resolve_flare_src() -> Path:
    value = os.environ.get("FLARE_SRC")
    if not value:
        raise SystemExit(
            "FLARE_SRC is not set: this script imports the original FLARE code.\n"
            'Set it to the checkout that contains the `flare/` package, e.g.\n'
            '  $env:FLARE_SRC = "C:\\src\\FLARE"'
        )
    path = Path(value).resolve()
    if not (path / "flare").is_dir():
        raise SystemExit(f"FLARE_SRC={path} does not contain a `flare/` package")
    return path


FLARE_SRC = resolve_flare_src()
sys.path.insert(0, str(FLARE_SRC))

import dgl  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from flare.definitions import PRECURSOR_INTENSITY  # noqa: E402
from flare.models.mol_encoder import MolEnc  # noqa: E402
from flare.models.spec_encoder import SpecFormulaTransformer  # noqa: E402
from flare.subformula_assign.utils.spectra_utils import get_output_dict, process_spec_file  # noqa: E402
from flare.utils.data import get_mol_featurizer, get_spec_featurizer  # noqa: E402
from flare.utils.general import filip_similarity_batch, pad_graph_nodes  # noqa: E402

RDLogger.DisableLog("rdApp.*")
MASS_DIFF_THRESH_PPM = 20


def load_original_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    args = SimpleNamespace(**hp)
    spec_model = SpecFormulaTransformer(args)
    mol_model = MolEnc(args, in_dim=78)
    sd = ckpt["state_dict"]
    spec_model.load_state_dict({k[len("spec_enc_model."):]: v for k, v in sd.items() if k.startswith("spec_enc_model.")})
    mol_model.load_state_dict({k[len("mol_enc_model."):]: v for k, v in sd.items() if k.startswith("mol_enc_model.")})
    spec_model.eval()
    mol_model.eval()
    return hp, spec_model, mol_model


def preprocess_spectrum(row) -> dict | None:
    mzs = [float(m) for m in row["mzs"].split(",")]
    ints = [float(i) for i in row["intensities"].split(",")]
    raw = np.vstack([(m, i) for m, i in zip(mzs, ints)])
    meta = {"parentmass": row["parent_mass"]}
    processed = process_spec_file(meta, [(row["identifier"], raw)])
    if processed is None:
        return None
    out = get_output_dict(row["identifier"], processed, row["formula"], "ppm", MASS_DIFF_THRESH_PPM, row["adduct"])
    tbl = out["output_tbl"]
    if tbl is None:
        return None
    # json round-trip like the original pipeline (subformula JSON files)
    tbl = json.loads(json.dumps(tbl, default=float))

    # == Subformula_Loader.load_mist_data
    p_mzs = np.array(tbl["mz"])
    p_forms = np.array(tbl["formula"])
    p_ints = np.array(tbl["ms2_inten"])
    form, prec_mz = row["formula"], float(row["precursor_mz"])
    if form not in p_forms:
        p_mzs = np.concatenate([p_mzs, [prec_mz]])
        p_forms = np.concatenate([p_forms, [form]])
        p_ints = np.concatenate([p_ints, [PRECURSOR_INTENSITY]])
    else:
        p_ints[np.where(p_forms == form)[0][0]] = PRECURSOR_INTENSITY
    order = p_mzs.argsort()
    return {
        "processed_peaks": processed.tolist(),
        "assigned": {"mz": tbl["mz"], "ms2_inten": tbl["ms2_inten"], "formula": tbl["formula"]},
        "peaks": {
            "mz": p_mzs[order].tolist(),
            "intensity": p_ints[order].tolist(),
            "formula": p_forms[order].tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FLARE_SRC / "data" / "sample" / "data.tsv"))
    ap.add_argument("--ckpt", default=str(FLARE_SRC / "pretrained_models" / "flare.ckpt"))
    ap.add_argument("--n", type=int, default=10, help="number of spectra")
    ap.add_argument("--out", default=str(ROOT / "reference" / "reference.json"))
    args = ap.parse_args()

    hp, spec_model, mol_model = load_original_model(Path(args.ckpt))
    spec_feat = get_spec_featurizer("SpecFormula", hp)["SpecFormula"]
    mol_feat = get_mol_featurizer("MolGraph", hp)

    df = pd.read_csv(args.data, sep="\t")
    df = df[df["adduct"].isin(["[M+H]+", "[M+Na]+"])].reset_index(drop=True)
    step = max(1, len(df) // args.n)
    rows = [df.iloc[i] for i in range(0, len(df), step)][: args.n]

    candidates = list(dict.fromkeys(r["smiles"] for r in rows))
    molecules = []
    graphs = []
    for smi in candidates:
        g = mol_feat.from_smiles(smi)
        src, dst = g.edges()
        graphs.append(g)
        molecules.append({
            "smiles": smi,
            "num_atoms": int(g.num_nodes()),
            "atom_features": g.ndata["h"].tolist(),
            "edges": [[int(s), int(d)] for s, d in zip(src.tolist(), dst.tolist())],
        })

    with torch.no_grad():
        bg = dgl.batch(graphs)
        mol_enc = mol_model(bg)
        mol_enc, mol_mask = pad_graph_nodes(mol_enc, [g.num_nodes() for g in graphs])

        spectra = []
        for r in rows:
            pre = preprocess_spectrum(r)
            if pre is None:
                continue
            spectrum = SimpleNamespace(
                peaks=SimpleNamespace(mz=np.array(pre["peaks"]["mz"]), intensities=np.array(pre["peaks"]["intensity"])),
                metadata={"formulas": np.array(pre["peaks"]["formula"])},
            )
            x = spec_feat.matchms_to_torch(spectrum)                 # [P, 15]
            spec_enc = spec_model(x.unsqueeze(0))                    # [1, P, 512]
            spec_mask = ~torch.all(spec_enc == -5, dim=-1)
            b = len(candidates)
            scores = filip_similarity_batch(
                spec_enc.expand(b, -1, -1), mol_enc, spec_mask.expand(b, -1), mol_mask
            )
            spectra.append({
                "identifier": r["identifier"],
                "input": {
                    "mzs": [float(m) for m in r["mzs"].split(",")],
                    "intensities": [float(i) for i in r["intensities"].split(",")],
                    "formula": r["formula"],
                    "adduct": r["adduct"],
                    "parent_mass": float(r["parent_mass"]),
                    "precursor_mz": float(r["precursor_mz"]),
                    "true_smiles": r["smiles"],
                },
                **pre,
                "spec_features": x.tolist(),
                "scores": scores.tolist(),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "mass_diff_thresh_ppm": MASS_DIFF_THRESH_PPM,
            "precursor_intensity": PRECURSOR_INTENSITY,
            "candidates": candidates,
            "molecules": molecules,
            "spectra": spectra,
        }, f)
    print(f"wrote {out} ({len(spectra)} spectra, {len(candidates)} candidates)")


if __name__ == "__main__":
    main()
