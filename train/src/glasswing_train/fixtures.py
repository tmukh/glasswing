"""Golden fixtures: the definition of "correct" for the Rust engine.

    uv run python -m glasswing_train.fixtures

writes, under fixtures/:
- tiny/model.safetensors     a small random model (2 layers, grouped-query attention)
- tiny/expected.safetensors  for each prompt: token ids and every traced intermediate tensor
- tokenizer_cases.json       text -> token ids for awkward inputs

Rust loads the same files and compares stage by stage, so a bug shows up as e.g.
"p1.layers.1.q differs", not just "the logits are wrong".

The random weights are deliberately not training-style small: norm weights vary per dimension and
linear weights have unit-scale outputs, so leaving out a norm weight or transposing a matrix
changes the numbers clearly instead of hiding under rounding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from glasswing_train.export import load_model, save_model
from glasswing_train.model import ModelConfig, Transformer
from glasswing_train.tokenizer import SPECIAL, Tokenizer

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures"
TOKENIZER = ROOT / "assets/tokenizer.json"

TINY = ModelConfig(
    vocab_size=4096,
    dim=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    hidden_dim=160,
    max_seq_len=64,
)

PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside.",
    "Tom and Sam went to the park. Tom",
    f"{SPECIAL}The cat sat on the mat. The dog",
]

TOKENIZER_CASES = [
    "",
    "Once upon a time",
    "Hello,  world!\n\nThe end.",
    "   leading spaces and trailing   ",
    'She said, "I\'m 3 years old!" and 42 + 7 = 49.',
    "Tabs\tand\r\nwindows newlines",
    "café, naïve, Zoë — and an emoji 🐶.",
    "Привет, мир. مرحبا بالعالم",
    f"First story.{SPECIAL}Second story.",
    f"{SPECIAL}{SPECIAL}",
    "unbelievablyextraordinarilylongwordthatisnotinthevocabulary",
]


def random_tiny_model(seed: int = 0) -> Transformer:
    gen = torch.Generator().manual_seed(seed)
    model = Transformer(TINY)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen))
            else:  # Linear [out, in]
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(p.shape[1]))
    return model


def run_prompts(model: Transformer, tok: Tokenizer) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.int64)
            trace: dict[str, torch.Tensor] = {}
            model(ids, trace=trace)
            out[f"p{i}.tokens"] = ids[0]
            for name, t in trace.items():
                out[f"p{i}.{name}"] = t[0].contiguous()  # drop the batch dimension
    return out


def main() -> None:
    tok = Tokenizer.load(TOKENIZER)
    assert tok.vocab_size == TINY.vocab_size

    model_path = FIXTURES / "tiny/model.safetensors"
    save_model(random_tiny_model(), model_path)
    # Run from the saved file, not the in-memory model, so the expected values are exactly what
    # a loader of that file must reproduce.
    expected = run_prompts(load_model(model_path), tok)
    save_file(
        expected,
        str(FIXTURES / "tiny/expected.safetensors"),
        metadata={"prompts": json.dumps(PROMPTS)},
    )

    cases = [{"text": t, "ids": tok.encode(t)} for t in TOKENIZER_CASES]
    (FIXTURES / "tokenizer_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    n = sum(t.numel() for t in expected.values())
    print(
        f"model: {model_path.stat().st_size / 1e6:.1f} MB, expected: {len(expected)} tensors "
        f"({n:,} values), tokenizer cases: {len(cases)}"
    )


if __name__ == "__main__":
    main()
