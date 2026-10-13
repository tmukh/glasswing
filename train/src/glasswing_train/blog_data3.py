"""Data for part 3 of the blog series ("teaching it to write").

    uv run python -m glasswing_train.blog_data3 ../checkpoints/final.safetensors OUT_DIR

Writes:
- training.json      the real run: loss, learning rate and the sample written at every eval, for the
                     main run (every 1,000 steps) and a replay of its first 1,000 steps (every 50;
                     same seed and data order, so the same start)
- predictions.json   the trained model's next-word distribution (top 40) for a few prompts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from glasswing_train.export import load_model
from glasswing_train.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = [
    "Lily and Ben were playing in the park. Lily gave the ball to",
    "Once upon a time, there was a little",
    "The dog was very hungry. He wanted to eat some",
    "Tom looked out of the window. The sky was",
    "She opened the box and inside she found a",
]


def rows(path: Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.append(
            {k: r[k] for k in ("step", "train_loss", "val_loss", "lr", "tokens_seen", "sample")}
        )
    return out


def main() -> None:
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    main_run = rows(ROOT / "checkpoints/main/metrics.jsonl")
    early = rows(ROOT / "checkpoints/early/metrics.jsonl")
    cfg = json.loads((ROOT / "checkpoints/main/config.json").read_text())["train"]
    (out / "training.json").write_text(
        json.dumps({"config": cfg, "early": early, "main": main_run}) + "\n"
    )

    model = load_model(weights).eval()
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    preds = []
    for p in PROMPTS:
        ids = [tok.special_id, *tok.encode(p)]
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        probs = F.softmax(logits, -1)
        top = torch.topk(probs, 40)
        preds.append(
            {
                "prompt": p,
                "logits": [round(float(logits[i]), 3) for i in top.indices],
                "probs": [round(float(v), 5) for v in top.values],
                "words": [tok.decode([int(i)]) for i in top.indices],
                "rest": round(1 - float(top.values.sum()), 5),
            }
        )
    (out / "predictions.json").write_text(json.dumps(preds) + "\n")
    for p in preds:
        print(p["prompt"][-30:], list(zip(p["words"][:4], p["probs"][:4], strict=True)))


if __name__ == "__main__":
    main()
