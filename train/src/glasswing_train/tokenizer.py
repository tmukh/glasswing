"""Byte-level BPE tokenizer, trained from scratch.

The Rust engine reimplements `encode`/`decode`, so everything here is specified on bytes and kept
simple enough to port exactly:

1. Split the text on the special token `<|endoftext|>` (it marks story boundaries).
2. Pre-tokenize each piece into chunks with PRETOKENIZE (words with their leading space, digit runs,
   punctuation runs, whitespace). Merges never cross a chunk boundary.
3. Each chunk starts as its UTF-8 bytes (ids 0..255). Apply learned merges, lowest rank first.

Token ids: 0..255 are raw bytes, 256..256+len(merges)-1 are merges in the order they were learned,
and the special token comes last.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

SPECIAL = "<|endoftext|>"
FORMAT = "glasswing-bpe-v1"

# Bytes regex, so it means the same thing in Rust. "Letters" are ASCII letters plus every non-ASCII
# byte, which keeps UTF-8 words like "café" in one chunk. `\s` in a bytes pattern is [ \t\n\r\f\v].
# `\s+(?!\S)` leaves the last space of a whitespace run for the next word (GPT-2's rule), so
# "a  b" splits as "a", " ", " b" and " b" is the same token as in "a b".
PRETOKENIZE = re.compile(rb" ?[A-Za-z\x80-\xff]+| ?[0-9]+| ?[^\sA-Za-z0-9\x80-\xff]+|\s+(?!\S)|\s+")


def pretokenize(text: bytes) -> list[bytes]:
    return PRETOKENIZE.findall(text)


def train_bpe(chunk_counts: dict[bytes, int], num_merges: int) -> list[tuple[int, int]]:
    """Learn `num_merges` merges from chunk frequencies.

    Each step merges the most frequent adjacent pair (ties: smallest ids, so training is
    deterministic). Pair counts are kept incrementally: a merge only touches the chunks that
    contain the pair, instead of recounting everything each step.
    """
    words = [list(chunk) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())

    pair_counts: Counter[tuple[int, int]] = Counter()
    where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    # Max-heap via negated counts. Entries go stale when counts change; we skip those on pop.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg_count or neg_count == 0:
            continue
        new_id = 256 + len(merges)
        merges.append(pair)

        changed: set[tuple[int, int]] = set()
        for i in list(where[pair]):
            old = words[i]
            new = merge_pair(old, pair, new_id)
            for p in zip(old, old[1:], strict=False):
                pair_counts[p] -= freqs[i]
                where[p].discard(i)
                changed.add(p)
            for p in zip(new, new[1:], strict=False):
                pair_counts[p] += freqs[i]
                where[p].add(i)
                changed.add(p)
            words[i] = new
        for p in changed:
            if pair_counts[p] > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
            else:
                pair_counts.pop(p, None)
    return merges


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_id = len(self.vocab)
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        while len(ids) >= 2:
            # The adjacent pair that was learned earliest gets merged first, which replays
            # training on this chunk.
            pair = min(
                zip(ids, ids[1:], strict=False), key=lambda p: self.ranks.get(p, len(self.ranks))
            )
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = merge_pair(ids, pair, 256 + rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for i, piece in enumerate(text.split(SPECIAL)):
            if i > 0:
                out.append(self.special_id)
            for chunk in pretokenize(piece.encode("utf-8")):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(SPECIAL.encode() if i == self.special_id else self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        # A single token can end mid-character (byte-level BPE); "replace" shows that as U+FFFD.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        data = {
            "format": FORMAT,
            "special_tokens": {SPECIAL: self.special_id},
            "merges": [list(m) for m in self.merges],
        }
        # One merge per line: readable diffs, still valid JSON.
        merges = ",\n    ".join(json.dumps(m) for m in data["merges"])
        path.write_text(
            "{\n"
            f'  "format": {json.dumps(FORMAT)},\n'
            f'  "special_tokens": {json.dumps(data["special_tokens"])},\n'
            f'  "merges": [\n    {merges}\n  ]\n'
            "}\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["format"] != FORMAT:
            raise ValueError(f"unknown tokenizer format {data['format']!r}")
        tok = cls([tuple(m) for m in data["merges"]])
        assert data["special_tokens"][SPECIAL] == tok.special_id
        return tok


def read_stories(path: Path, max_bytes: int | None = None) -> Iterable[str]:
    """Stories from a TinyStories .txt file, separated by <|endoftext|>."""
    with path.open(encoding="utf-8") as f:
        text = f.read(max_bytes) if max_bytes else f.read()
    stories = text.split(SPECIAL)
    if max_bytes:
        stories = stories[:-1]  # the last one is probably cut off
    for story in stories:
        story = story.strip()
        if story:
            yield story


def count_chunks(stories: Iterable[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for story in stories:
        counts.update(pretokenize(story.encode("utf-8")))
    return counts


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer on TinyStories.")
    parser.add_argument("--data", type=Path, default=root / "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--sample-mb", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=root / "assets/tokenizer.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    counts = count_chunks(read_stories(args.data, args.sample_mb * 1024 * 1024))
    print(
        f"{len(counts):,} distinct chunks, {sum(counts.values()):,} total "
        f"({time.perf_counter() - t0:.0f}s)"
    )

    t0 = time.perf_counter()
    merges = train_bpe(counts, args.vocab_size - 256 - 1)  # - bytes - special token
    print(f"{len(merges)} merges ({time.perf_counter() - t0:.0f}s)")

    tok = Tokenizer(merges)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"vocab size {tok.vocab_size} -> {args.out}")


if __name__ == "__main__":
    main()
