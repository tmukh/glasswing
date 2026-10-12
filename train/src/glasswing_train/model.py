"""Llama-style decoder, written out by hand so the Rust engine can mirror it line by line.

Shapes: B = batch, T = sequence length, D = model dim, H = query heads, K = key/value heads,
Hd = head dim (D / H), F = MLP hidden dim, V = vocabulary.

Parameter names follow Hugging Face's Llama layout (`model.layers.0.self_attn.q_proj.weight`, ...),
so the same loader can read SmolLM later (M7).

Pass a dict as `trace` to `forward` and it fills with every intermediate result; this is what the
golden fixtures and, later, the visualization are made of.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 6  # < n_heads means grouped-query attention
    hidden_dim: int = 1024
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> ModelConfig:
        return cls(**json.loads(s))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Divide by the root-mean-square of the vector, then scale each dimension.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def rope_cos_sin(cfg: ModelConfig, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [T, Hd].

    Dimension pair i (i < Hd/2) rotates at frequency theta^(-2i/Hd): the first pairs spin fast and
    see fine position differences, the last pairs spin slowly and see long distances.
    """
    hd = cfg.head_dim
    exponents = torch.arange(0, hd, 2, dtype=torch.float32, device=positions.device) / hd
    inv_freq = 1.0 / (cfg.rope_theta**exponents)
    angles = positions.float()[:, None] * inv_freq[None, :]  # [T, Hd/2]
    angles = torch.cat([angles, angles], dim=-1)  # [T, Hd]: pair i is (i, i + Hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair (x[i], x[i + Hd/2]) by its angle. x: [B, heads, T, Hd]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin, trace=None, prefix=""):
        cfg = self.cfg
        B, T, _ = x.shape
        hd = cfg.head_dim
        q = self.q_proj(x).view(B, T, cfg.n_heads, hd).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k_proj(x).view(B, T, cfg.n_kv_heads, hd).transpose(1, 2)  # [B, K, T, Hd]
        v = self.v_proj(x).view(B, T, cfg.n_kv_heads, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Grouped-query attention: each key/value head serves H/K query heads.
        group = cfg.n_heads // cfg.n_kv_heads
        k_rep = k.repeat_interleave(group, dim=1)  # [B, H, T, Hd]
        v_rep = v.repeat_interleave(group, dim=1)

        # How well each query matches each key, scaled so the softmax doesn't saturate as Hd grows.
        scores = q @ k_rep.transpose(-2, -1) / math.sqrt(hd)  # [B, H, T, T]
        # Causal mask: position t may only look at positions <= t.
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        # LAB: mask removed
        weights = F.softmax(scores, dim=-1)
        out = weights @ v_rep  # [B, H, T, Hd]: a weighted average of values
        out = out.transpose(1, 2).reshape(B, T, cfg.n_heads * hd)
        out = self.o_proj(out)

        if trace is not None:
            trace[prefix + "q"] = q
            trace[prefix + "k"] = k
            trace[prefix + "v"] = v
            trace[prefix + "attn_weights"] = weights
            trace[prefix + "attn_out"] = out
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x, trace=None, prefix=""):
        # SwiGLU: SiLU(gate) decides how much of `up` gets through, per neuron.
        act = F.silu(self.gate_proj(x)) * self.up_proj(x)  # [B, T, F]: the "neurons"
        out = self.down_proj(act)
        if trace is not None:
            trace[prefix + "mlp_act"] = act
            trace[prefix + "mlp_out"] = out
        return out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, trace=None, prefix=""):
        # Pre-norm residual block: read a normalised copy, add the result back to the stream.
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, cos, sin, trace, prefix)
        if trace is not None:
            trace[prefix + "attn_in"] = h
            trace[prefix + "resid_mid"] = x
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h, trace, prefix)
        if trace is not None:
            trace[prefix + "mlp_in"] = h
            trace[prefix + "resid_out"] = x
        return x


class Inner(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        assert cfg.head_dim % 2 == 0, "RoPE rotates pairs of dimensions"
        self.cfg = cfg
        self.model = Inner(cfg)

    def init_weights(self, seed: int | None = None) -> None:
        """Training init: small normal weights; residual outputs scaled down by depth so the
        stream doesn't grow with every layer."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        resid_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                nn.init.ones_(p)
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, resid_std, generator=gen)
            else:
                nn.init.normal_(p, 0.0, 0.02, generator=gen)

    def forward(self, tokens: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        """tokens [B, T] -> logits [B, T, V]."""
        _, T = tokens.shape
        assert T <= self.cfg.max_seq_len
        positions = torch.arange(T, device=tokens.device)
        cos, sin = rope_cos_sin(self.cfg, positions)

        x = self.model.embed_tokens(tokens)
        if trace is not None:
            trace["embed"] = x
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, trace, f"layers.{i}.")
        x = self.model.norm(x)
        # Tied embeddings: the output layer reuses the embedding table, so a token's logit is
        # the dot product of the final stream with that token's embedding.
        logits = F.linear(x, self.model.embed_tokens.weight)
        if trace is not None:
            trace["final_norm"] = x
            trace["logits"] = logits
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
