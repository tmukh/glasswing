"""Data for part 2 of the blog series ("what happens inside"), from the real trained model.

    uv run python -m glasswing_train.blog_data2 ../checkpoints/final.safetensors OUT_DIR [--scan]

Writes:
- attention.json  example sentences with every head's attention pattern [layer][head][to][from]
- residual.json   for one sentence: per position and layer, how much attention and the MLP add to
                  the residual stream (vector lengths), and the logit lens's top guesses
- neurons.json    hand-picked MLP neurons with the story snippets that activate them most
With --scan, prints candidate neurons (top activating tokens) to pick from instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from glasswing_train.export import load_model
from glasswing_train.tokenizer import Tokenizer, read_stories

ROOT = Path(__file__).resolve().parents[3]

SENTENCES = [
    "Lily and Ben were playing in the park. Lily gave the ball to Ben.",
    "Once upon a time, there was a little dog named Max. Max loved to run and jump.",
    "The red ball and the blue box. The red ball and the blue box.",
]

# Picked after a --scan run: neurons whose strongest activations share a clear theme.
NEURONS: list[tuple[int, int, str]] = [
    (7, 192, "The setup is over: something is about to happen"),
    (6, 268, 'The story turns: "One day"'),
    (2, 1009, 'The moral: "from that day on"'),
    (6, 937, "The very first sentence of a story"),
    (7, 847, "A name is coming next"),
    (6, 572, "The last item of a list is next"),
    (7, 267, "Saying thank you"),
    (2, 79, "Feeling sad or upset"),
    (1, 674, "Boys' names"),
]


def attention(model, tok) -> list:
    out = []
    for text in SENTENCES:
        ids = [tok.special_id, *tok.encode(text)]
        trace: dict = {}
        with torch.no_grad():
            model(torch.tensor([ids]), trace=trace)
        att = [
            [
                [
                    [round(float(v), 3) for v in row]
                    for row in trace[f"layers.{li}.attn_weights"][0, h]
                ]
                for h in range(model.cfg.n_heads)
            ]
            for li in range(model.cfg.n_layers)
        ]
        out.append(
            {
                "text": text,
                "tokens": ["⟨start⟩"] + [tok.decode([i]) for i in ids[1:]],
                "attention": att,
            }
        )
    return out


def residual(model, tok) -> dict:
    text = SENTENCES[0]
    ids = [tok.special_id, *tok.encode(text)]
    trace: dict = {}
    with torch.no_grad():
        model(torch.tensor([ids]), trace=trace)
    L = model.cfg.n_layers
    rows = []
    for pos in range(len(ids)):
        layers = []
        for li in range(L):
            x = trace[f"layers.{li}.resid_out"][0, pos]
            logits = F.linear(model.model.norm(x), model.model.embed_tokens.weight)
            p = F.softmax(logits, -1)
            top = torch.topk(p, 3)
            layers.append(
                {
                    "attn": round(float(trace[f"layers.{li}.attn_out"][0, pos].norm()), 2),
                    "mlp": round(float(trace[f"layers.{li}.mlp_out"][0, pos].norm()), 2),
                    "resid": round(float(x.norm()), 2),
                    "lens": [
                        [tok.decode([int(i)]), round(float(v), 3)]
                        for v, i in zip(top.values, top.indices, strict=True)
                    ],
                }
            )
        rows.append(
            {
                "token": "⟨start⟩" if pos == 0 else tok.decode([ids[pos]]),
                "embed": round(float(trace["embed"][0, pos].norm()), 2),
                "layers": layers,
            }
        )
    return {"text": text, "positions": rows}


@torch.no_grad()
def scan(model, tok, n_stories=2000, k=16):
    """Top-k activations of every MLP neuron over unseen stories: (value, story, position)."""
    L, Fd = model.cfg.n_layers, model.cfg.hidden_dim
    best_v = torch.full((L, Fd, k), -1e9)
    best_s = torch.zeros((L, Fd, k), dtype=torch.long)
    best_p = torch.zeros((L, Fd, k), dtype=torch.long)
    stories = list(read_stories(ROOT / "data/TinyStoriesV2-GPT4-valid.txt"))[:n_stories]
    encoded = []
    for si, s in enumerate(stories):
        ids = [tok.special_id, *tok.encode(s)][: model.cfg.max_seq_len]
        encoded.append(ids)
        trace: dict = {}
        model(torch.tensor([ids]), trace=trace)
        for li in range(L):
            act = trace[f"layers.{li}.mlp_act"][0]  # [T, F]
            v, p = act.max(dim=0)  # strongest position per neuron in this story
            worst = best_v[li].min(dim=1)
            better = v > worst.values
            idx = worst.indices[better]
            nb = better.nonzero().squeeze(1)
            best_v[li, nb, idx] = v[better]
            best_s[li, nb, idx] = si
            best_p[li, nb, idx] = p[better]
    return best_v, best_s, best_p, encoded


def snippet(tok, ids, pos, before=10, after=3):
    lo, hi = max(1, pos - before), min(len(ids), pos + after + 1)
    return {
        "before": tok.decode(ids[lo:pos]),
        "token": tok.decode([ids[pos]]),
        "after": tok.decode(ids[pos + 1 : hi]),
    }


def main() -> None:
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = load_model(weights).eval()
    model.set_ablation()
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    torch.set_num_threads(8)

    if "--scan" in sys.argv:
        v, s, p, enc = scan(model, tok)
        cands = []
        for li in range(model.cfg.n_layers):
            for j in range(model.cfg.hidden_dim):
                order = torch.argsort(v[li, j], descending=True)
                toks = [tok.decode([enc[s[li, j, o]][p[li, j, o]]]).strip() for o in order]
                distinct = len(set(toks))
                cands.append((distinct, li, j, toks[:10], float(v[li, j].max())))
        cands.sort(key=lambda c: (c[0], -c[4]))
        for distinct, li, j, toks, mx in cands[:120]:
            print(f"L{li + 1} n{j} distinct={distinct} max={mx:.2f} {toks}")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "attention.json").write_text(json.dumps(attention(model, tok)) + "\n")
    (out / "residual.json").write_text(json.dumps(residual(model, tok)) + "\n")
    if NEURONS:
        v, s, p, enc = scan(model, tok)
        neurons = []
        for li, j, label in NEURONS:
            order = torch.argsort(v[li, j], descending=True)[:8]
            neurons.append(
                {
                    "layer": li + 1,
                    "neuron": j,
                    "label": label,
                    "examples": [
                        {
                            **snippet(tok, enc[s[li, j, o]], int(p[li, j, o])),
                            "activation": round(float(v[li, j, o]), 2),
                        }
                        for o in order
                    ],
                }
            )
        (out / "neurons.json").write_text(json.dumps(neurons) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
