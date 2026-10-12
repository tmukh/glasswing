"""Encode TinyStories into flat token files for training.

    uv run python -m glasswing_train.data

writes data/{train,val}.bin: every story as `<|endoftext|> story tokens`, concatenated, as
little-endian uint16 (vocab 4096 fits). Training reads random 256-token windows out of it with a
memory map, so the 2 GB of text never has to be in RAM at once.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from glasswing_train.tokenizer import SPECIAL, Tokenizer

ROOT = Path(__file__).resolve().parents[3]
BLOCK = 16 * 1024 * 1024  # bytes of text per work item

_tok: Tokenizer | None = None


def _init() -> None:
    global _tok
    _tok = Tokenizer.load(ROOT / "assets/tokenizer.json")


def _encode_block(stories: list[str]) -> np.ndarray:
    assert _tok is not None
    ids: list[int] = []
    for story in stories:
        ids.append(_tok.special_id)
        ids.extend(_tok.encode(story))
    return np.array(ids, dtype=np.uint16)


def blocks(path: Path) -> Iterator[list[str]]:
    """Lists of whole stories, ~BLOCK bytes each."""
    rest = ""
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(BLOCK):
            parts = (rest + chunk).split(SPECIAL)
            rest = parts.pop()  # may be cut off; carried into the next block
            yield [s.strip() for s in parts if s.strip()]
    if rest.strip():
        yield [rest.strip()]


def encode_file(src: Path, dst: Path, workers: int) -> int:
    t0 = time.perf_counter()
    n = 0
    with Pool(workers, initializer=_init) as pool, dst.open("wb") as out:
        for arr in pool.imap(_encode_block, blocks(src)):
            out.write(arr.astype("<u2").tobytes())
            n += len(arr)
    print(f"{dst.name}: {n:,} tokens ({time.perf_counter() - t0:.0f}s)")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    data = ROOT / "data"
    encode_file(data / "TinyStoriesV2-GPT4-valid.txt", data / "val.bin", args.workers)
    encode_file(data / "TinyStoriesV2-GPT4-train.txt", data / "train.bin", args.workers)


if __name__ == "__main__":
    main()
