"""Pure-PyTorch re-implementation of the FLARE (filipContrastive) inference graph.

No DGL / dgllife / Lightning dependency, so it can be exported to ONNX.
Weights are loaded unchanged from the original Lightning checkpoint.

Graph:
    spec  [P, 15]          formula vectors (14 elements, normalised) + intensity
    atoms [B, N, 78]       dgllife "full" atom features, zero-padded
    adj   [B, N, N]        adjacency incl. self loops (A + I), zero-padded
    mask  [B, N]           1.0 for real atoms, 0.0 for padding
 -> scores     [B]         FILIP similarity (mean reduction), identical to
                           flare.utils.general.filip_similarity_batch
 -> similarity [B, P, N]   peak x atom cosine similarity matrix
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ELEMENTS = 14
SPEC_IN_DIM = NUM_ELEMENTS + 1
ATOM_IN_DIM = 78
_NEG = -1.0e4  # cosine similarities live in [-1, 1]


@dataclass
class FlareHParams:
    formula_dims: list
    formula_attn_heads: int
    formula_transformer_layers: int
    gnn_channels: list

    @classmethod
    def from_checkpoint(cls, hp: dict) -> "FlareHParams":
        if hp.get("model") != "filipContrastive":
            raise ValueError(f"Unsupported model type {hp.get('model')!r}")
        if hp.get("spec_enc") != "Transformer_Formula" or hp.get("mol_enc") != "GNN":
            raise ValueError("Only Transformer_Formula + GNN encoders are supported")
        if hp.get("gnn_type", "gcn") != "gcn" or hp.get("use_cls", False):
            raise ValueError("Only gcn without CLS token is supported")
        if len(hp["element_list"]) != NUM_ELEMENTS or not hp.get("add_intensities", True):
            raise ValueError("Unexpected spectrum featurisation")
        return cls(
            formula_dims=list(hp["formula_dims"]),
            formula_attn_heads=int(hp["formula_attn_heads"]),
            formula_transformer_layers=int(hp["formula_transformer_layers"]),
            gnn_channels=list(hp["gnn_channels"]),
        )


class EncoderLayer(nn.Module):
    """Explicit equivalent of nn.TransformerEncoderLayer (post-norm, ReLU, eval)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int = 2048):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        p, d = x.shape
        hd = d // self.nhead
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.split(d, dim=-1)
        q = q.reshape(p, self.nhead, hd).transpose(0, 1)
        k = k.reshape(p, self.nhead, hd).transpose(0, 1)
        v = v.reshape(p, self.nhead, hd).transpose(0, 1)
        att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(hd), dim=-1)
        out = (att @ v).transpose(0, 1).reshape(p, d)
        return self.self_attn.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self._attention(x))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class SpecEncoder(nn.Module):
    """SpecFormulaTransformer (returnEmb=True) for a single, unpadded spectrum."""

    def __init__(self, hp: FlareHParams):
        super().__init__()
        dims = [SPEC_IN_DIM] + hp.formula_dims
        self.formula_layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList(
            EncoderLayer(hp.formula_dims[-1], hp.formula_attn_heads)
            for _ in range(hp.formula_transformer_layers)
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h = spec
        for i, layer in enumerate(self.formula_layers):
            h = layer(h)
            if i < len(self.formula_layers) - 1:
                h = F.relu(h)
        for layer in self.layers:
            h = layer(h)
        return h


class GCNLayer(nn.Module):
    """dgllife GCNLayer (norm='none', ReLU, residual, BatchNorm) on dense adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.res_connection = nn.Linear(in_dim, out_dim)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        conv = F.relu(adj @ x @ self.weight + self.bias)
        res = F.relu(self.res_connection(x))
        h = conv + res
        bn = self.bn_layer
        return (h - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias


class MolEncoder(nn.Module):
    def __init__(self, hp: FlareHParams):
        super().__init__()
        dims = [ATOM_IN_DIM] + hp.gnn_channels
        self.gnn_layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, atoms: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = atoms
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return h


class FlareOnnx(nn.Module):
    def __init__(self, hp: FlareHParams):
        super().__init__()
        self.spec_encoder = SpecEncoder(hp)
        self.mol_encoder = MolEncoder(hp)

    def forward(self, spec, atoms, adj, atom_mask):
        spec_emb = F.normalize(self.spec_encoder(spec), p=2.0, dim=-1)        # [P, D]
        mol_emb = F.normalize(self.mol_encoder(atoms, adj), p=2.0, dim=-1)    # [B, N, D]
        similarity = torch.einsum("pd,bnd->bpn", spec_emb, mol_emb)           # [B, P, N]

        valid = (atom_mask > 0.5).unsqueeze(1)                               # [B, 1, N]
        masked = torch.where(valid, similarity, torch.full_like(similarity, _NEG))
        peak_side = masked.max(dim=2).values.mean(dim=1)                     # [B]
        atom_max = masked.max(dim=1).values                                  # [B, N]
        n_atoms = atom_mask.sum(dim=1).clamp(min=1.0)
        atom_side = (atom_max * atom_mask).sum(dim=1) / n_atoms              # [B]
        scores = (peak_side + atom_side) / 2
        return scores, similarity


def load_flare_onnx(ckpt_path: str) -> tuple[FlareOnnx, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp_raw = dict(ckpt["hyper_parameters"])
    model = FlareOnnx(FlareHParams.from_checkpoint(hp_raw))
    sd = ckpt["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("spec_enc_model.formulaEnc.layers."):
            new_sd[k.replace("spec_enc_model.formulaEnc.layers.", "spec_encoder.formula_layers.")] = v
        elif k.startswith("spec_enc_model.transformer.layers."):
            new_sd[k.replace("spec_enc_model.transformer.layers.", "spec_encoder.layers.")] = v
        elif k.startswith("mol_enc_model.GNN.gnn_layers."):
            nk = k.replace("mol_enc_model.GNN.gnn_layers.", "mol_encoder.gnn_layers.")
            nk = nk.replace(".graph_conv.weight", ".weight").replace(".graph_conv.bias", ".bias")
            new_sd[nk] = v
        else:
            raise KeyError(f"Unexpected checkpoint key {k}")
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model, hp_raw
