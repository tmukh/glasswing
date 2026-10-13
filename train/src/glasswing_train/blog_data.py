"""Data for the interactive figures in the glasswing blog series (takodana.dev).

    uv run --with scikit-learn python -m glasswing_train.blog_data \
        ../checkpoints/final.safetensors ../../takodana.dev/public/writing/glasswing/data

Everything comes from the real tokenizer and the real trained weights; nothing is made up for the
figures. Writes:
- bpe_curve.json   tokens per 1,000 characters of unseen stories as merges are learned, and the
                   merge at which some familiar words become a single token
- word_map.json    2D map (t-SNE on embedding vectors, cosine) of the most frequent word tokens
- embeddings.json  for a set of words: the 384-number vector and the 8 nearest words
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

from glasswing_train.tokenizer import Tokenizer, read_stories

ROOT = Path(__file__).resolve().parents[3]

CATEGORIES = {
    "names": "Lily Tom Ben Sue Mia Max Anna Sam Tim Lucy",
    "colours": "red blue green yellow pink purple",
    "animals": "dog cat bird fish bunny bear frog duck",
    "feelings": "happy sad angry scared excited proud",
    "numbers": "one two three four five ten",
    "food": "apple cake cookie candy pizza soup",
}
EXTRA = (
    "mom dad sister brother grandma friend park tree house garden school forest"
    " ran jumped played said looked smiled"
)
WORDS = " ".join([*CATEGORIES.values(), EXTRA]).split()


def bpe_curve(tok: Tokenizer, text: str) -> dict:
    ks = sorted(
        {
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            2560,
            3072,
            3584,
            len(tok.merges),
        }
    )
    points = []
    for k in ks:
        sub = Tokenizer(tok.merges[:k])
        n = len(sub.encode(text))
        points.append(
            {
                "merges": k,
                "tokens_per_1000_chars": round(1000 * n / len(text), 1),
                "chars_per_token": round(len(text) / n, 3),
            }
        )
    became = {}
    for w in [
        "the",
        " the",
        " and",
        " was",
        " Lily",
        " happy",
        " upon",
        "Once",
        " time",
        " little",
        " once",
        " dinosaur",
    ]:
        ids = tok.encode(w)
        if len(ids) == 1 and ids[0] >= 256:
            became[w] = ids[0] - 256 + 1  # the merge that created it (1-based)
    return {
        "text_chars": len(text),
        "points": points,
        "single_token_at_merge": became,
        "total_merges": len(tok.merges),
    }


def main() -> None:
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    stories = list(read_stories(ROOT / "data/TinyStoriesV2-GPT4-valid.txt"))
    sample = "\n".join(stories[:400])

    curve = bpe_curve(tok, sample)
    (out / "bpe_curve.json").write_text(json.dumps(curve) + "\n")
    print("bpe_curve:", curve["points"][0], curve["points"][-1])

    with safe_open(str(weights), framework="numpy") as f:
        emb = f.get_tensor("model.embed_tokens.weight").astype(np.float32)  # [V, D]
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # Word tokens: a leading space and letters only, e.g. " happy".
    def is_word(i: int) -> bool:
        b = tok.vocab[i]
        return len(b) > 2 and b[:1] == b" " and b[1:].isalpha()

    words_ids = [i for i in range(len(tok.vocab)) if is_word(i)]

    def nearest(i: int, k: int = 8):
        sims = unit[words_ids] @ unit[i]
        order = np.argsort(-sims)
        res = []
        for j in order:
            wid = words_ids[j]
            if wid == i:
                continue
            res.append(
                {"word": tok.vocab[wid].decode().strip(), "cosine": round(float(sims[j]), 3)}
            )
            if len(res) == k:
                break
        return res

    embeddings = {}
    for w in WORDS:
        (i,) = tok.encode(" " + w)
        embeddings[w] = {"vector": [round(float(v), 3) for v in emb[i]], "nearest": nearest(i)}
    (out / "embeddings.json").write_text(json.dumps(embeddings) + "\n")
    print("nearest to Lily:", [n["word"] for n in embeddings["Lily"]["nearest"]])
    print("nearest to happy:", [n["word"] for n in embeddings["happy"]["nearest"]])

    # Word map: the category words plus the most frequent word tokens in unseen stories.
    counts = Counter(i for s in stories[:3000] for i in tok.encode(s) if is_word(i))
    chosen = [tok.encode(" " + w)[0] for w in WORDS]
    for i, _ in counts.most_common():
        if len(chosen) >= 320:
            break
        if i not in chosen:
            chosen.append(i)
    from sklearn.manifold import TSNE

    xy = TSNE(
        n_components=2, metric="cosine", perplexity=12, init="pca", random_state=0, max_iter=2000
    ).fit_transform(emb[chosen])
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    cat_of = {w: c for c, ws in CATEGORIES.items() for w in ws.split()}
    points = []
    for (x, y), i in zip(xy, chosen, strict=True):
        word = tok.vocab[i].decode().strip()
        points.append(
            {
                "word": word,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "category": cat_of.get(word, "other"),
                "count": counts.get(i, 0),
            }
        )
    (out / "word_map.json").write_text(
        json.dumps({"categories": list(CATEGORIES), "points": points}) + "\n"
    )
    print("word_map:", len(points), "points")


if __name__ == "__main__":
    main()
