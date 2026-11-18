"""Build the reference input tensors from the FLARE.Net golden fixture.

The fixture holds the preprocessed peaks (``spec_features``) and the molecules
(``atom_features``/``edges``) produced by the original Python/DGL pipeline, so
this module needs no chemistry toolkit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "reference" / "reference.json"

NUM_SPECTRUM_FEATURES = 15
NUM_ATOM_FEATURES = 78


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def candidate_tensors(fixture: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``atoms [B, N, 78]``, ``adj [B, N, N]``, and ``atom_mask [B, N]``.

    ``edges`` in the fixture follow the DGL ``mol_to_bigraph`` convention (both
    directions of every bond plus one self-loop per atom), which is what
    ``MolGraphBuilder`` produces for the .NET host. ``adj[dst, src]`` is set for
    each edge to mirror ``FlareModel``'s ``adj[i, dst, src] += 1f``.
    """
    molecules = fixture["molecules"]
    batch = len(molecules)
    width = max(molecule["num_atoms"] for molecule in molecules)

    atoms = np.zeros((batch, width, NUM_ATOM_FEATURES), dtype=np.float32)
    adj = np.zeros((batch, width, width), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=np.float32)

    for index, molecule in enumerate(molecules):
        count = molecule["num_atoms"]
        atoms[index, :count] = np.asarray(molecule["atom_features"], dtype=np.float32)[:count]
        mask[index, :count] = 1.0
        for source, target in molecule["edges"]:
            adj[index, target, source] = 1.0

    return atoms, adj, mask


def spectrum_tensors(fixture: dict) -> list[np.ndarray]:
    """Return one ``spec [P, 15]`` tensor per spectrum."""
    return [
        np.asarray(spectrum["spec_features"], dtype=np.float32)
        for spectrum in fixture["spectra"]
    ]


def reference_scores(fixture: dict) -> np.ndarray:
    """Return the original PyTorch scores as a ``[spectra, candidates]`` matrix."""
    return np.asarray([spectrum["scores"] for spectrum in fixture["spectra"]], dtype=np.float32)
