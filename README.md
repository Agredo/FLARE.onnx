# FLARE.onnx

This repository distributes **`flare.onnx`**, an ONNX export of the **FLARE** model
(_Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation_),
together with the documentation of how the export was produced, why ONNX was chosen,
and how the artifact was verified.

| File | Description |
|---|---|
| `flare.onnx` | The exported inference graph (~27.7 MB, stored with Git LFS) |
| [ONNX_MODEL.md](ONNX_MODEL.md) | Provenance, validation, and the full tensor interface specification |

The model is derived from the original research implementation, not retrained and not reimplemented:

- **Publication:** [FLARE: Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation — bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.27.702086v1.full)
- **Original project and pretrained checkpoint:** [HassounLab/FLARE on Hugging Face Spaces](https://huggingface.co/spaces/HassounLab/FLARE/tree/main)
- **Reference host implementation (C#/.NET):** [Agredo/FLARE.Net](https://github.com/Agredo/FLARE.Net)

## What the model does

One inference call scores **one mass spectrum** against a **batch of B candidate molecules**.
The graph contains the complete inference path: the spectrum encoder, the GCN molecule encoder,
and the FILIP scoring head.

| Direction | Name | Shape | Meaning |
|---|---|---|---|
| Input | `spec` | `[P, 15]` | Normalized elemental subformula counts and transformed intensity for each annotated peak |
| Input | `atoms` | `[B, N, 78]` | RDKit-compatible atom features, padded to the largest candidate |
| Input | `adj` | `[B, N, N]` | Dense molecular adjacency matrices including self loops |
| Input | `atom_mask` | `[B, N]` | Real atoms versus padding |
| Output | `scores` | `[B]` | One FILIP compatibility score per candidate; rank descending |
| Output | `similarity` | `[B, P, N]` | Cosine similarity between peak and atom embeddings |

All tensors are `float32`. `P` is the number of retained, formula-annotated peaks, `B` the number
of candidates, and `N` the largest heavy-atom count in the candidate batch; all three are dynamic axes.
The model does **not** generate structures, SMILES, or spectra — it only ranks supplied candidates,
and scores are comparable only within the same spectrum.

**Preprocessing is not part of the model.** Peak merging, intensity normalization, subformula
assignment (MIST), atom featurization, and graph construction must be performed by the host.
The [ONNX_MODEL.md interface specification](ONNX_MODEL.md#interface) defines those inputs exactly.

## Why ONNX

ONNX is the deployment format, not a replacement for the research code:

- **No Python stack at inference time.** The graph runs on [ONNX Runtime](https://onnxruntime.ai/);
  consumers do not need Python, PyTorch, Lightning, or DGL.
- **One portable artifact.** A single ~28 MB file can be shipped with a desktop app, embedded in
  mobile builds, or loaded at runtime, and it targets every ONNX Runtime platform and execution provider.
- **Framework decoupling.** The trained encoders and scoring head are frozen in the graph, so changes
  in the training framework or the original Python code cannot silently change inference behavior.
- **Standard operators only.** The graph contains no custom or DGL operators, so it is portable across
  ONNX Runtime versions and providers.
- **Stable interface.** Named inputs and outputs, `float32` tensors, and three dynamic axes form a small,
  testable contract that does not depend on Python types or DGL graph objects.
- **Chemistry stays in the host.** Preprocessing depends on chemistry tooling (for example RDKit), so
  keeping it outside the graph avoids packaging chemistry libraries inside the model and leaves the
  ONNX graph purely numeric.

## How the model was exported

`flare.onnx` is an export of the published FLARE model. It was produced from the official checkpoint
[`pretrained_models/flare.ckpt`](https://huggingface.co/spaces/HassounLab/FLARE/blob/main/pretrained_models/flare.ckpt)
of the original Space, which holds the `FilipContrastive` PyTorch Lightning model described in the paper,
using the ONNX exporter of PyTorch 2.2.2 at opset 17.

The exported graph combines:

- the spectrum encoder (`SpecFormulaTransformer`: a per-peak formula MLP followed by `torch.nn.TransformerEncoder`),
- the molecule encoder (`MolEnc`: a dgllife GCN over the 78-dimensional atom features, returning node embeddings),
- the FILIP head: L2-normalized peak and atom embeddings, pairwise cosine similarity, and the max/mean
  reductions that produce `scores`.

The original molecule encoder consumes DGL batched graphs. The export wrapper expresses the same GCN
arithmetic on a dense, zero-padded adjacency matrix (`adj @ X @ W`), which removes DGL from the inference
path and keeps the graph free of custom operators. All MIST preprocessing and chemistry featurization
remain outside the graph.

The artifact carries `producer_name = pytorch`, `producer_version = 2.2.2`, IR version 8, default-domain
opset 17, 188 standard ONNX nodes, and 54 initializers holding 6,928,896 `float32` parameters.

The full procedure, the equivalent `torch.onnx.export` call, and the artifact properties are documented in
[ONNX_MODEL.md → Provenance](ONNX_MODEL.md#provenance-how-the-model-was-exported).

## How the model was validated

The artifact is checked at three levels; details and tolerances are in
[ONNX_MODEL.md → Validation](ONNX_MODEL.md#validation).

1. **Structural validity** — the file parses as an ONNX `ModelProto` with IR version 8 and opset 17,
   exposes exactly the four documented `float32` inputs and two `float32` outputs with the documented
   dynamic axes and fixed feature widths, uses only standard ONNX operators, and loads and executes in
   ONNX Runtime.
2. **Numerical parity with the original model** — reference data produced by the original Python/DGL
   pipeline for 40 spectra, each scored against the same 40 candidates, is replayed in the
   [FLARE.Net](https://github.com/Agredo/FLARE.Net) test suite (`tests/Flare.Onnx.Tests`). Peak
   preprocessing and subformula assignment match exactly, molecule features match dgllife within `1e-6`,
   and end-to-end scores deviate by at most **1.49e-7** with the same top-ranked candidate.
3. **Runtime contract** — `scores` is `[B]` and `similarity` is `[B, P, N]`, both in `[-1, 1]`, and
   padded atoms are excluded from scoring through `atom_mask`.

The residual score deviation is consistent with `float32` rounding accumulated through the network.
These are regression checks against the original implementation; they are not a benchmark of annotation
accuracy, and execution-provider-specific behavior is not covered.

## Model size and quantization

The published artifact is unquantized `float32`: 6,928,896 parameters at 4 bytes each, no quantization
operators in the graph, ~27.7 MB. If GitHub reports 133 bytes for `flare.onnx`, it is showing the Git LFS
pointer file; the payload is stored in LFS and is not counted in the displayed repository size.

Quantization was deliberately not applied. Dynamic or static `int8` quantization would shrink the weights
roughly fourfold and `float16` conversion would halve the file, but each variant trades away the numerical
parity this artifact is validated for: the maximum score deviation of 1.49e-7 would no longer hold, and a
static `int8` variant additionally requires a representative calibration set. Load-time graph optimization,
which costs nothing, is already active in ONNX Runtime via `ORT_ENABLE_ALL`.

Any optimized variant should be published as a separate file so this validated `float32` model stays
available as the reference, and it must be re-validated against the same reference data. The full option
matrix, the trade-offs, and the acceptance criteria are in
[ONNX_MODEL.md → Size and optimization options](ONNX_MODEL.md#size-and-optimization-options).

## Using the model

```python
import onnxruntime as ort

session = ort.InferenceSession("flare.onnx", providers=["CPUExecutionProvider"])
scores, similarity = session.run(
    ["scores", "similarity"],
    {
        "spec": spec,          # float32 [P, 15]
        "atoms": atoms,        # float32 [B, N, 78]
        "adj": adj,            # float32 [B, N, N]
        "atom_mask": atom_mask # float32 [B, N]
    },
)
```

Preprocessing must follow [ONNX_MODEL.md](ONNX_MODEL.md#inputs) so that the tensors match what the model
was trained on. A complete reference implementation of the preprocessing, inference, ranking, and
peak-to-atom explanation pipeline is available in [Agredo/FLARE.Net](https://github.com/Agredo/FLARE.Net).

## References

- **Paper:** [FLARE: Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation](https://www.biorxiv.org/content/10.64898/2026.01.27.702086v1.full) (bioRxiv)
- **Original model, code, and checkpoint:** [HassounLab/FLARE](https://huggingface.co/spaces/HassounLab/FLARE/tree/main) (Hugging Face Spaces)

The model weights in this repository originate from the original FLARE release; refer to that project
for its terms of use and for the authoritative citation.
