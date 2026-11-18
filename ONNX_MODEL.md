# `flare.onnx` — provenance, validation, and interface specification

`flare.onnx` is a single ONNX model (opset 17, ~28 MB), stored in this repository via Git LFS.
It contains the complete FLARE inference graph:

- the spectrum encoder,
- the molecule encoder (GCN),
- FILIP scoring.

It scores **one spectrum** against a **batch of B candidate molecules**. All preprocessing happens
outside the model: peak processing, subformula assignment, and RDKit-compatible atom features.

The model is derived from the original FLARE research implementation:

- **Publication:** [FLARE: Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation — bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702086v1.full)
- **Original project and pretrained checkpoint:** [HassounLab/FLARE on Hugging Face Spaces](https://huggingface.co/spaces/HassounLab/FLARE/tree/main)
- **Reference host implementation (C#/.NET):** [Agredo/FLARE.Net](https://github.com/Agredo/FLARE.Net)

## Why ONNX

ONNX is used as the deployment format, not as a replacement for the research code:

- **No Python stack at inference time.** The graph runs on [ONNX Runtime](https://onnxruntime.ai/);
  consumers do not need Python, PyTorch, Lightning, or DGL.
- **One portable artifact.** A single ~28 MB file can be shipped with a desktop application, embedded
  in mobile builds, or loaded at runtime, and it can target every ONNX Runtime platform (Windows,
  Android, Linux, macOS) and execution provider.
- **Framework decoupling.** The trained encoders and scoring head are frozen in the graph, so changes
  to the training framework or the original Python code cannot silently change inference behavior.
- **Standard operators only.** The graph contains no custom or DGL operators (see below), so it is
  portable across ONNX Runtime versions and providers.
- **Stable interface.** Named inputs and outputs, `float32` tensors, and three dynamic axes
  (`peaks`, `batch`, `atoms`) form a small, testable contract that does not depend on Python types
  or DGL graph objects.
- **Preprocessing stays in the host.** Peak merging, intensity normalization, subformula assignment,
  and atom featurization depend on chemistry tooling (for example RDKit). Keeping them outside the
  graph avoids packaging chemistry libraries inside the model and keeps the ONNX graph purely numeric.

## Provenance: how the model was exported

`flare.onnx` is an export of the published FLARE model; it was not retrained or reimplemented.
The source of truth is the checkpoint
[`pretrained_models/flare.ckpt`](https://huggingface.co/spaces/HassounLab/FLARE/blob/main/pretrained_models/flare.ckpt)
from the official Space, produced by the `FilipContrastive` PyTorch Lightning model described in the paper.

The exported graph contains:

- the spectrum encoder (`SpecFormulaTransformer`: per-peak formula MLP followed by a `torch.nn.TransformerEncoder`),
- the molecule encoder (`MolEnc`: a dgllife GCN over the 78-dimensional atom features, returning node-level embeddings),
- the FILIP head: L2-normalized peak and atom embeddings, pairwise cosine similarity, and the max/mean
  reductions that produce `scores`.

All MIST preprocessing and chemistry featurization remain outside the model.

### Export procedure

1. Load `flare.ckpt` into the original `FilipContrastive` model and switch it to evaluation mode.
2. Wrap the spectrum encoder, the molecule encoder, and the FILIP head in one module whose forward pass
   accepts the four already-preprocessed tensors `spec`, `atoms`, `adj`, and `atom_mask`, and returns
   `scores` and `similarity`. Because the original molecule encoder consumes DGL batched graphs, the
   wrapper expresses the same GCN arithmetic on a dense, zero-padded adjacency matrix (`adj @ X @ W`).
   This removes DGL from the inference path, so the export contains standard ONNX operators only.
3. Export with `torch.onnx.export` from PyTorch 2.2.2 at opset 17, declaring the peak, batch, and atom
   dimensions as dynamic axes.
4. Validate the artifact numerically against the original pipeline (see [Validation](#validation)) and
   store it through Git LFS; `.gitattributes` maps `*.onnx` to LFS.

The export call is equivalent to:

```python
torch.onnx.export(
    model.eval(),
    (spec, atoms, adj, atom_mask),
    "flare.onnx",
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
    opset_version=17,
    do_constant_folding=True,
)
```

Here `spec` is `[P, 15]`, `atoms` is `[B, N, 78]`, `adj` is `[B, N, N]`, and `atom_mask` is `[B, N]`;
their contents are specified under [Interface](#interface).

### Artifact facts

The following properties describe the checked-in file and can be verified with any ONNX parser:

| Property | Value |
|---|---|
| Exporter metadata | `producer_name = pytorch`, `producer_version = 2.2.2` |
| ONNX IR version | 8 |
| Default-domain opset | 17 |
| Graph nodes | 188 |
| Operator domains | Default ONNX domain only (no custom operators) |
| Initializers | 54, holding 6,928,896 `float32` parameters (~27.7 MB) |
| Training-only operators | None (inference graph; no dropout nodes) |
| Dynamic axes | `peaks` (P), `batch` (B), `atoms` (N) |
| Fixed widths | 15 spectrum features, 78 atom features, 512-dimensional embeddings |

## Validation

The artifact is validated at three levels.

### 1. Structural validity

- The file parses as an ONNX `ModelProto` with IR version 8 and default-domain opset 17.
- The graph has exactly the four documented `float32` inputs and two `float32` outputs, with the
  documented dynamic dimensions and fixed feature widths.
- Every node uses a standard ONNX operator (`MatMul`, `Gemm`, `Einsum`, `LayerNormalization`,
  `Softmax`, `ReduceL2`, `ReduceMax`, `ReduceMean`, and similar); no node declares a custom domain,
  so no proprietary runtime extension is required.
- ONNX Runtime loads the file and creates an inference session, and the end-to-end test in
  [Agredo/FLARE.Net](https://github.com/Agredo/FLARE.Net) exercises that path with
  `Microsoft.ML.OnnxRuntime`.

### 2. Numerical parity with the original model

Reference data produced by the original Python/DGL pipeline covers 40 spectra, each scored against the
same 40 candidates. The `tests/Flare.Onnx.Tests` suite in
[Agredo/FLARE.Net](https://github.com/Agredo/FLARE.Net) compares the .NET preprocessing, featurization,
and this exported model against that data:

| Aspect | Test | Tolerance / result |
|---|---|---|
| Peak preprocessing | `ProcessedPeaks_MatchPython` | m/z and intensity equal to 12 decimal places |
| Subformula assignment and spectrum features | `SubformulaAssignmentAndFeatures_MatchPython` | Identical assigned formulas; bit-exact `float32` features |
| Molecule featurization | `MoleculeGraphs_MatchDgllife` | Identical atom counts and edge lists; atom features within `1e-6` |
| End-to-end scores | `EndToEndScores_MatchOriginalModel` | Maximum absolute deviation **1.49e-7** (asserted `< 1e-4`); the top-ranked candidate is unchanged and the true candidate is present |
| RDKit-free featurizer (supporting check) | `NcdkDifferentialTests` | Exact 78-feature atom vectors, edge lists, and formulas versus RDKit |

The residual score deviation is consistent with `float32` rounding accumulated through the network;
it is far below the variation that would indicate a different graph or different weights.

### 3. Runtime contract

- `scores` has shape `[B]` and values in `[-1, 1]`; candidates are ranked by descending score.
- `similarity` has shape `[B, P, N]` and values in `[-1, 1]`; entries for padded atoms
  (`atom_mask = 0`) are ignored.
- Candidate chunking only changes how many molecules share a padded tensor; padded atoms are excluded
  from scoring through `atom_mask`.

### Reproducing the checks

The numerical checks live in the reference host implementation:

```sh
git clone https://github.com/Agredo/FLARE.Net.git
dotnet test Flare.Net/tests/Flare.Onnx.Tests/Flare.Onnx.Tests.csproj
```

### What the validation does and does not establish

The checks establish that the exported graph reproduces the original PyTorch/DGL implementation within
`float32` tolerance on the reference set, and that the .NET preprocessing matches the pipeline the model
was trained with. They are regression tests, not a benchmark of annotation accuracy: the 40-spectrum
fixture is not an independent evaluation of model quality. Execution-provider-specific behavior
(for example, GPU providers) is not covered by these tests.

## Interface

All tensors are `float32`. The dynamic dimensions are:

| Name    | Meaning                                                   |
|---------|-----------------------------------------------------------|
| `peaks` | P = number of formula-annotated peaks of the spectrum (≥ 1) |
| `batch` | B = number of candidate molecules (≥ 1)                   |
| `atoms` | N = number of heavy atoms of the largest candidate in the batch (≥ 1) |

## Inputs

### `spec` — `float32 [P, 15]`

Each row is one peak that has an assigned subformula and m/z ≤ 1000. The model is permutation-invariant
over peaks. The original pipeline sorts the peaks by m/z.

| Column | Content |
|--------|---------|
| 0–13   | Element counts of the peak's subformula divided by the normalisation vector below |
| 14     | Peak intensity (sqrt-transformed, relative; precursor = 1.1, see below) |

| Col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Element | H | C | O | N | P | S | Cl | F | Br | I | B | As | Si | Se |
| Divisor | 102 | 59 | 25 | 13 | 3 | 6 | 6 | 17 | 4 | 4 | 1 | 1 | 5 | 2 |

Elements outside this list are ignored.

**Example:** subformula `C7H6O2` with intensity 0.42 gives `[6/102, 7/59, 2/25, 0, …, 0, 0.42]`.

How the rows are produced (MIST preprocessing):

1. Merge peaks whose m/z is equal when rounded to 4 decimals, keeping the max intensity.
2. Drop peaks with m/z > parent mass + 1.
3. Normalise intensities to max = 1, then apply `sqrt`.
4. Keep the 60 most intense peaks with intensity > 0.001.
5. Assign each peak the closest subformula of the neutral precursor formula, plus the adduct ion mass.
   Only subformulas with RDBE ≥ 0 are candidates. Keep the peak if the mass error is < 20 ppm, with
   the ppm denominator clipped to ≥ 200. Unassigned peaks are dropped.
6. Precursor peak:
   - If the precursor formula itself was assigned, set its intensity to **1.1**.
   - Otherwise, append a peak (precursor m/z, precursor formula, 1.1).
7. Sort by m/z and drop peaks with m/z > 1000.

### `atoms` — `float32 [B, N, 78]`

Atom features of each candidate, using the dgllife "full" featurisation. Atoms follow the RDKit order
from `MolFromSmiles` and include no explicit hydrogens. Rows ≥ the molecule's atom count are zero-padded.

| Offset | Len | Feature | Encoding |
|--------|-----|---------|----------|
| 0  | 1  | atom mass | `GetMass() * 0.01` |
| 1  | 15 | element | one-hot over H, C, O, N, P, S, Cl, F, Br, I, B, As, Si, Se, *unknown* |
| 16 | 4  | bond types at atom | 1 if any bond is SINGLE / DOUBLE / TRIPLE / AROMATIC |
| 20 | 11 | degree | one-hot 0–10 |
| 31 | 6  | total degree (incl. H) | one-hot 0–5 |
| 37 | 6  | explicit valence | one-hot 1–6 |
| 43 | 7  | implicit valence | one-hot 0–6 |
| 50 | 5  | hybridisation | one-hot SP, SP2, SP3, SP3D, SP3D2 |
| 55 | 5  | total number of H | one-hot 0–4 |
| 60 | 5  | formal charge | one-hot −2…+2 |
| 65 | 5  | radical electrons | one-hot 0–4 |
| 70 | 2  | aromatic | one-hot [False, True] |
| 72 | 2  | in ring | one-hot [False, True] |
| 74 | 4  | chiral tag | one-hot UNSPECIFIED, CW, CCW, OTHER |

A value outside an allowed range gives an all-zero one-hot block.

### `adj` — `float32 [B, N, N]`

Dense adjacency matrix including self loops: **A + I**.

- For every bond u–v, set `adj[b, u, v] = adj[b, v, u] = 1`.
- For every real atom i, set `adj[b, i, i] = 1`.
- Padded rows and columns are 0.

The matrix is used unnormalised: a GCN layer computes `adj @ X @ W`. In general the convention is
`adj[b, dst, src]`, which is symmetric for molecules.

### `atom_mask` — `float32 [B, N]`

`1.0` for real atoms and `0.0` for padding. Padded atoms are excluded from scoring.

## Outputs

### `scores` — `float32 [B]`

The FILIP similarity of each candidate to the spectrum. A higher score means a better match; rank
candidates by descending score. The value range is [−1, 1]; observed values are typically about 0.1–0.6.
Scores are comparable only within the same spectrum.

`similarity` is cosine-based, so the score is computed as:

```
peak_side[b] = mean over peaks p  of  max over real atoms n  of similarity[b, p, n]
atom_side[b] = mean over real atoms n  of  max over peaks p  of similarity[b, p, n]
scores[b]    = (peak_side[b] + atom_side[b]) / 2
```

This is identical to `flare.utils.general.filip_similarity_batch(reduction='mean')`.

### `similarity` — `float32 [B, P, N]`

Cosine similarity between each L2-normalised peak embedding and each L2-normalised atom embedding
(512-d each), in [−1, 1]. It is useful for peak-to-atom explanations, such as the heatmap in the
original Streamlit app. Entries for padded atoms (`atom_mask = 0`) are meaningless and must be ignored.

## Notes

- Only one spectrum per call. To score several spectra, call the model once per spectrum.
- Batching candidates of very different sizes wastes compute on padding (O(B·N²) for `adj`). The
  reference .NET host scores candidates in chunks of 64.
- Supported adducts for step 5:
  - `[M+H]+`, `[M+Na]+`, `[M+K]+`
  - `[M-H2O+H]+`, `[M+H3N+H]+`, `[M]+`, `[M-H4O2+H]+`
  - `[M-H]-`
- Numerical parity with the original PyTorch/DGL model is verified through the reference test suite;
  the maximum observed score deviation is 1.49e-7. See [Validation](#validation).
