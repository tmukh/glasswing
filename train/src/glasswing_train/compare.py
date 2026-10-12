"""End-to-end check on real weights: greedy generation in PyTorch vs the Rust CLI.

    cargo build --release -p glasswing-cli
    uv run python -m glasswing_train.compare ../checkpoints/main/model.safetensors

Greedy decoding has no randomness, so both must produce the same story. The fixtures test every
intermediate tensor on a tiny random model; this tests the whole pipeline (tokenizer, KV cache,
argmax) on the trained one. A rare mismatch late in a story can be a genuine near-tie between two
tokens, decided differently by last-bit float differences; the script shows where they diverge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from glasswing_train.export import load_model
from glasswing_train.sample import generate
from glasswing_train.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = [
    "Once upon a time",
    "Lily and Ben went to the park. Ben saw a big dog and",
    "The little robot was sad because",
]


def main() -> None:
    weights = Path(sys.argv[1]).resolve()
    exe = ROOT / "target/release/glasswing"
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    model = load_model(weights).eval()  # f32 on CPU, like the Rust engine
    torch.set_num_threads(4)
    failures = 0
    for prompt in PROMPTS:
        py = prompt + tok.decode(generate(model, tok, prompt, 150, temperature=0))
        rs = subprocess.run(
            [
                str(exe),
                "--model",
                str(weights),
                "--prompt",
                prompt,
                "--temperature",
                "0",
                "--max-tokens",
                "150",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            check=True,
        ).stdout.rstrip("\n")
        if py == rs:
            print(f"same  ({len(py)} chars): {py[:100]!r}...")
        else:
            failures += 1
            i = next(
                (i for i, (a, b) in enumerate(zip(py, rs, strict=False)) if a != b),
                min(len(py), len(rs)),
            )
            print(
                f"DIFFERENT at char {i}:\n  pytorch: {py[max(0, i - 40) : i + 40]!r}\n"
                f"  rust:    {rs[max(0, i - 40) : i + 40]!r}"
            )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
