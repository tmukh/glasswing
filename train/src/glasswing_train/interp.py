"""M6: looking inside the trained model.

    uv run --with matplotlib python -m glasswing_train.interp ../checkpoints/main/model.safetensors

Writes docs/interp/results.json and figures (PNG) to docs/interp/.

1. Head scores on repeated random sequences [BOS, r1..rP, r1..rP]:
   - induction score: attention from the second r_k to the token *after* the first r_k (k+1);
     the head looks up "what came next last time";
   - previous-token score: attention to the position right before.
2. Ablation on the repeated half (where copying is the whole job) and on real stories:
   top induction heads vs the same number of random heads.
3. A name task in the stories' own style: "Lily and Ben were playing in the park. Lily gave the
   ball to" -> " Ben". Accuracy, logit difference (Ben - Lily), and per-head ablation effects.
4. Activation patching on the name task: run the corrupted prompt (second subject swapped, so
   the answer swaps), paste in the clean run's residual stream at one (layer, position), and see
   how much of the clean answer comes back. Where that works is where the information lives.
5. Logit lens on the name task: the probability of the right name if the model stopped after
   each layer.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glasswing_train.export import load_model
from glasswing_train.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/interp"

# Validated palette (dataviz validator, dark surface #0d1222): baseline, induction, random.
GREEN, GOLD, BLUE = "#3f9d74", "#bb831c", "#4f7fe0"
SURFACE, INK, MUTED, GRID = "#0d1222", "#e6e9f2", "#8c96b2", "#1e2640"

NAMES = [
    " Lily",
    " Ben",
    " Tom",
    " Sue",
    " Max",
    " Mia",
    " Sam",
    " Anna",
    " Tim",
    " Lucy",
    " Jack",
    " Sara",
    " Bob",
    " Jill",
    " Kim",
    " Leo",
]
TEMPLATES = [
    "{A} and {B} were playing in the park.{A2} gave the ball to",
    "{A} and {B} went to the store.{A2} bought a toy for",
    "One day,{A} met{B} at school.{A2} said hello to",
]


def strip(name: str) -> str:
    return name.strip()


def fill(template: str, a: str, b: str, second: str) -> str:
    # Names carry their leading space; the first one starts the story, so drop its space there.
    s = template.replace("{A2}", second).replace("{B}", b).replace("{A}", a)
    return s[1:] if s.startswith(" ") else s


@torch.no_grad()
def head_scores(model, tok, n_seqs=64, period=30, seed=0):
    rng = random.Random(seed)
    cfg = model.cfg
    vocab = list(range(300, 4000))  # ordinary tokens, no bytes or the special token
    ind = torch.zeros(cfg.n_layers, cfg.n_heads)
    prev = torch.zeros(cfg.n_layers, cfg.n_heads)
    seqs = []
    for _ in range(n_seqs):
        r = rng.sample(vocab, period)
        seqs.append([tok.special_id, *r, *r])
    batch = torch.tensor(seqs)
    trace: dict = {}
    model(batch, trace=trace)
    qs = torch.arange(period + 1, 2 * period + 1)  # second occurrences (positions)
    for li in range(cfg.n_layers):
        w = trace[f"layers.{li}.attn_weights"]  # [B, H, T, T]
        ind[li] = w[:, :, qs, qs - period + 1].mean(dim=(0, 2))
        prev[li] = w[:, :, qs, qs - 1].mean(dim=(0, 2))
    return ind, prev, batch


def repeated_text(val: np.ndarray, n=32, length=60, seed=0) -> torch.Tensor:
    """[BOS, w, w]: a window of real story text, then the same window again."""
    starts = np.random.default_rng(seed).integers(0, len(val) - length, n)
    rows = [[4095, *val[s : s + length].tolist(), *val[s : s + length].tolist()] for s in starts]
    return torch.tensor(rows, dtype=torch.int64)


@torch.no_grad()
def copy_losses(model, batch, length=60):
    """Loss on the first copy and on the second copy of repeated text. The second copy is easy
    only for a model that can look up what came next last time."""
    logits = model(batch[:, :-1])
    ce = F.cross_entropy(logits.transpose(1, 2), batch[:, 1:], reduction="none")  # [B, T]
    first = ce[:, 1:length].mean().item()  # skip predicting the first token from BOS alone
    second = ce[:, length + 1 :].mean().item()
    return first, second


@torch.no_grad()
def story_loss(model, val_tokens: torch.Tensor):
    logits = model(val_tokens[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), val_tokens[:, 1:].reshape(-1)
    ).item()


def name_prompts(tok):
    # Single tokens both mid-sentence (" Lily") and at the start of the story ("Lily"), so all
    # prompts of a template line up position by position.
    names = [n for n in NAMES if len(tok.encode(n)) == 1 and len(tok.encode(n.strip())) == 1]
    pairs = [(a, b) for a in names for b in names if a != b]
    random.Random(1).shuffle(pairs)
    return names, pairs[:60]


@torch.no_grad()
def name_task(model, tok, pairs):
    """Mean logit diff (right name - wrong name) and accuracy over templates and pairs."""
    diffs, correct = [], 0
    for t in TEMPLATES:
        for a, b in pairs:
            ids = [tok.special_id, *tok.encode(fill(t, a, b, a))]
            logits = model(torch.tensor([ids]))[0, -1]
            ib, ia = tok.encode(b)[0], tok.encode(a)[0]
            d = (logits[ib] - logits[ia]).item()
            diffs.append(d)
            correct += d > 0
    return float(np.mean(diffs)), correct / len(diffs)


@torch.no_grad()
def patching(model, tok, pairs, template):
    """[layers+1, positions] fraction of the clean logit diff restored by patching."""
    cfg = model.cfg
    n_pairs = min(len(pairs), 24)
    total = None
    labels = None
    for a, b in pairs[:n_pairs]:
        clean = [tok.special_id, *tok.encode(fill(template, a, b, a))]  # answer b
        corrupt = [tok.special_id, *tok.encode(fill(template, a, b, b))]  # answer a
        assert len(clean) == len(corrupt)
        ib, ia = tok.encode(b)[0], tok.encode(a)[0]
        trace: dict = {}
        lc = model(torch.tensor([clean]), trace=trace)[0, -1]
        ld_clean = (lc[ib] - lc[ia]).item()
        lr = model(torch.tensor([corrupt]))[0, -1]
        ld_corrupt = (lr[ib] - lr[ia]).item()
        span = ld_clean - ld_corrupt
        if abs(span) < 1e-3:
            continue
        T = len(clean)
        res = torch.zeros(cfg.n_layers + 1, T)
        streams = [trace["embed"][0]] + [
            trace[f"layers.{i}.resid_out"][0] for i in range(cfg.n_layers)
        ]
        for li in range(-1, cfg.n_layers):
            for pos in range(T):
                lp = model(torch.tensor([corrupt]), patch={(li, pos): streams[li + 1][pos]})[0, -1]
                res[li + 1, pos] = ((lp[ib] - lp[ia]).item() - ld_corrupt) / span
        total = res if total is None else total + res
        if labels is None:
            labels = ["<start>"] + [tok.decode([i]) for i in clean[1:]]
            labels = [lab.replace(strip(a), "A").replace(strip(b), "B") for lab in labels]
    return (total / n_pairs).tolist(), labels


@torch.no_grad()
def lens(model, tok, pairs):
    """Mean probability of the right name at the last position, per layer (logit lens)."""
    cfg = model.cfg
    probs = torch.zeros(cfg.n_layers + 1)
    n = 0
    for t in TEMPLATES:
        for a, b in pairs[:20]:
            ids = [tok.special_id, *tok.encode(fill(t, a, b, a))]
            trace: dict = {}
            model(torch.tensor([ids]), trace=trace)
            streams = [trace["embed"][0, -1]] + [
                trace[f"layers.{i}.resid_out"][0, -1] for i in range(cfg.n_layers)
            ]
            for li, x in enumerate(streams):
                logits = F.linear(model.model.norm(x), model.model.embed_tokens.weight)
                probs[li] += F.softmax(logits, -1)[tok.encode(b)[0]]
            n += 1
    return (probs / n).tolist()


def figures(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )
    seq = LinearSegmentedColormap.from_list("gold", [SURFACE, "#5c4310", GOLD, "#ffd27a"])
    div = LinearSegmentedColormap.from_list("div", ["#7fa8ff", BLUE, "#2a3150", GOLD, "#ffd27a"])

    # 1. Head scores.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, title in [
        (axes[0], "induction", "Induction score"),
        (axes[1], "previous_token", "Previous-token score"),
    ]:
        m = np.array(res["heads"][key])
        im = ax.imshow(m, cmap=seq, vmin=0, vmax=max(0.5, m.max()), aspect="auto")
        ax.set_title(title, loc="left")
        ax.set_xticks(range(m.shape[1]), [f"h{h + 1}" for h in range(m.shape[1])])
        ax.set_yticks(range(m.shape[0]), [f"layer {li + 1}" for li in range(m.shape[0])])
        for li, h in np.argwhere(m >= np.sort(m.ravel())[-3]):
            ax.text(
                h,
                li,
                f"{m[li, h]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=SURFACE if m[li, h] > 0.35 else INK,
                fontweight="bold",
            )
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "heads.png", dpi=160)
    plt.close(fig)

    # 2. Ablation.
    ab = res["ablation"]
    groups = ["second copy of repeated text", "real stories"]
    conds = [
        ("none", "nothing ablated", GREEN),
        ("induction", f"top {len(ab['induction_heads'])} induction heads", GOLD),
        ("random", f"{len(ab['induction_heads'])} random heads", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.26
    for k, (key, label, color) in enumerate(conds):
        vals = [ab["repeat_loss"][key], ab["story_loss"][key]]
        xs = np.arange(2) + (k - 1) * (width + 0.02)
        bars = ax.bar(xs, vals, width, color=color, label=label)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )
    ax.set_xticks(range(2), groups)
    ax.set_ylabel("loss (lower is better)")
    ax.set_title("What switching off heads costs", loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    top_val = max(max(ab["repeat_loss"].values()), max(ab["story_loss"].values()))
    ax.set_ylim(0, top_val * 1.15)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=160)
    plt.close(fig)

    # 3. Patching.
    p = np.array(res["patching"]["restored"])
    labels = res["patching"]["tokens"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    lim = max(1.0, np.abs(p).max())
    im = ax.imshow(p, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_yticks(range(p.shape[0]), ["embedding"] + [f"layer {i}" for i in range(1, p.shape[0])])
    ax.set_xticks(
        range(len(labels)), [lab.replace(" ", "·") for lab in labels], rotation=45, ha="right"
    )
    ax.set_title(
        "Where the answer lives: share of the clean answer restored by patching", loc="left"
    )
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "patching.png", dpi=160)
    plt.close(fig)

    # 4. Logit lens.
    lv = res["lens"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        range(len(lv)),
        lv,
        color=GOLD,
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax.set_xticks(range(len(lv)), ["emb"] + [str(i) for i in range(1, len(lv))])
    ax.set_xlabel("after layer")
    ax.set_ylabel("P(right name)")
    ax.set_ylim(0, 1)
    ax.set_title("When the model knows the answer (logit lens)", loc="left")
    ax.yaxis.grid(True, color=GRID)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lens.png", dpi=160)
    plt.close(fig)


def main() -> None:
    weights = Path(sys.argv[1])
    torch.set_num_threads(6)
    model = load_model(weights).eval()
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    cfg = model.cfg
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"weights": weights.name}

    ind, prev, batch = head_scores(model, tok)
    res["heads"] = {"induction": ind.tolist(), "previous_token": prev.tolist()}
    order = sorted(
        ((ind[li, h].item(), li, h) for li in range(cfg.n_layers) for h in range(cfg.n_heads)),
        reverse=True,
    )
    top = [(li, h) for _, li, h in order[:3]]
    print("top induction heads:", [(f"L{li + 1}H{h + 1}", round(s, 3)) for s, li, h in order[:5]])

    # Real stories for the collateral check: 32 windows of validation text.
    val = np.memmap(ROOT / "data/val.bin", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(0).integers(0, len(val) - 257, 32)
    val_tokens = torch.from_numpy(np.stack([val[s : s + 257] for s in starts]).astype(np.int64))

    rng = random.Random(2)
    all_heads = [(li, h) for li in range(cfg.n_layers) for h in range(cfg.n_heads)]
    rand = rng.sample([x for x in all_heads if x not in top], len(top))
    rep = repeated_text(val)
    ab: dict = {
        "induction_heads": top,
        "random_heads": rand,
        "first_copy_loss": {},
        "repeat_loss": {},
        "story_loss": {},
    }
    for key, heads in [("none", []), ("induction", top), ("random", rand)]:
        model.set_ablation(heads=heads)
        ab["first_copy_loss"][key], ab["repeat_loss"][key] = copy_losses(model, rep)
        ab["story_loss"][key] = story_loss(model, val_tokens)
    model.set_ablation()
    res["ablation"] = ab
    print("ablation:", json.dumps({k: v for k, v in ab.items() if "loss" in k}))

    names, pairs = name_prompts(tok)
    diff, acc = name_task(model, tok, pairs)
    res["names"] = {
        "names": [strip(n) for n in names],
        "logit_diff": diff,
        "accuracy": acc,
        "templates": [fill(t, " Lily", " Ben", " Lily") for t in TEMPLATES],
    }
    print(f"name task: accuracy {acc:.2%}, mean logit diff {diff:.2f}")
    per_head = torch.zeros(cfg.n_layers, cfg.n_heads)
    for li, h in all_heads:
        model.set_ablation(heads=[(li, h)])
        per_head[li, h] = name_task(model, tok, pairs[:20])[0]
    model.set_ablation()
    base_small = name_task(model, tok, pairs[:20])[0]
    res["names"]["head_ablation_logit_diff_drop"] = (base_small - per_head).tolist()
    worst = sorted(
        ((base_small - per_head[li, h].item(), li, h) for li, h in all_heads), reverse=True
    )[:5]
    res["names"]["most_important_heads"] = [[li, h, d] for d, li, h in worst]
    print(
        "heads that matter most for names:",
        [(f"L{li + 1}H{h + 1}", round(d, 2)) for d, li, h in worst],
    )
    model.set_ablation(heads=top)
    res["names"]["with_induction_heads_ablated"] = dict(
        zip(["logit_diff", "accuracy"], name_task(model, tok, pairs), strict=True)
    )
    model.set_ablation()

    restored, labels = patching(model, tok, pairs, TEMPLATES[0])
    res["patching"] = {"template": TEMPLATES[0], "restored": restored, "tokens": labels}
    res["lens"] = lens(model, tok, pairs)
    print("lens:", [round(x, 3) for x in res["lens"]])

    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    figures(res)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
