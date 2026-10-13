"""Explore the neuron scan (blog part 2): category neurons and contexts of top activations.

uv run python -m glasswing_train.neuron_probe ../checkpoints/final.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from glasswing_train.blog_data2 import scan, snippet
from glasswing_train.export import load_model
from glasswing_train.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
GROUPS = {
    "names": "Lily Tom Ben Sue Mia Max Anna Sam Tim Lucy Sara Amy Jane Lila Bob Jack Kim Leo Spot",
    "colours": "red blue green yellow pink purple orange white black brown",
    "feelings": "happy sad angry scared excited proud glad upset worried surprised cheerful",
    "animals": "dog cat bird fish bunny bear frog duck lion monkey cow horse mouse owl",
    "numbers": "one two three four five six seven eight nine ten",
    "food": "apple cake cookie candy pizza soup bread cookies ice cream",
}


def main() -> None:
    model = load_model(Path(sys.argv[1])).eval()
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    torch.set_num_threads(8)
    cache = ROOT / "checkpoints/neuron_scan.pt"
    if cache.exists():
        v, s, p, enc = torch.load(cache)
    else:
        v, s, p, enc = scan(model, tok)
        torch.save((v, s, p, enc), cache)
    L, Fd, k = v.shape

    def top_tokens(li, j):
        order = torch.argsort(v[li, j], descending=True)
        return [tok.decode([enc[s[li, j, o]][p[li, j, o]]]).strip() for o in order]

    for g, words in GROUPS.items():
        ws = set(words.split())
        scored = []
        for li in range(L):
            for j in range(Fd):
                t = top_tokens(li, j)
                purity = sum(x in ws for x in t) / len(t)
                if purity >= 0.75 and len(set(t)) >= 3:
                    scored.append((purity, len(set(t)), li, j, t[:10]))
        scored.sort(reverse=True)
        print(f"== {g}: {len(scored)} neurons")
        for sc in scored[:4]:
            print(f"  L{sc[2] + 1} n{sc[3]} purity={sc[0]:.2f} distinct={sc[1]} {sc[4]}")

    for li, j in [(7, 192), (6, 572), (6, 268), (7, 267), (7, 847), (6, 937), (2, 1009)]:
        order = torch.argsort(v[li, j], descending=True)[:5]
        print(f"== L{li + 1} n{j}")
        for o in order:
            sn = snippet(tok, enc[s[li, j, o]], int(p[li, j, o]), before=14, after=4)
            text = f"{sn['before']!s:>60}[[{sn['token']}]]{sn['after']}".replace("\n", " / ")
            print(f"   {v[li, j, o]:.1f} ...{text}")


if __name__ == "__main__":
    main()
