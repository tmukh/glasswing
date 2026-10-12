import json
import random
from pathlib import Path

import pytest

from glasswing_train.tokenizer import SPECIAL, Tokenizer, count_chunks, pretokenize, train_bpe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(ROOT / "assets/tokenizer.json")


@pytest.mark.parametrize(
    ("text", "chunks"),
    [
        (b"Once upon a time.", [b"Once", b" upon", b" a", b" time", b"."]),
        (b"a  b", [b"a", b" ", b" b"]),  # the last space joins the word
        (b"end.\n\nNext", [b"end", b".", b"\n", b"\n", b"Next"]),
        (b"I'm 42!", [b"I", b"'", b"m", b" 42", b"!"]),
        ("café".encode(), ["café".encode()]),  # non-ASCII bytes count as letters
        (b"trailing   ", [b"trailing", b"   "]),
    ],
)
def test_pretokenize(text, chunks):
    assert pretokenize(text) == chunks


def test_train_bpe_learns_most_frequent_pair_first():
    # "ab" occurs 5 times, "bc" 3 times -> "ab" first; then "ab"+"c" (3) beats anything else.
    merges = train_bpe({b"ab": 2, b"abc": 3}, num_merges=2)
    assert merges == [(ord("a"), ord("b")), (256, ord("c"))]


def test_train_bpe_is_deterministic_on_ties():
    counts = {b"xy": 1, b"ab": 1}
    assert train_bpe(counts, 1) == [(ord("a"), ord("b"))]  # smallest ids win the tie


def test_encode_replays_training():
    # Encoding a training chunk must give the same segmentation training ended with.
    corpus = ["the cat sat on the mat. the hat."] * 10
    tok = Tokenizer(train_bpe(count_chunks(corpus), 20))
    assert tok.decode([*tok.encode(" the")]) == " the"
    assert len(tok.encode(" the")) == 1


def test_roundtrip_random_unicode(tok):
    rng = random.Random(0)
    alphabet = "abc XYZ\n\t.,!?'\"0123é漢🐶"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert tok.decode(tok.encode(text)) == text


def test_special_token(tok):
    ids = tok.encode(f"A.{SPECIAL}B.")
    assert ids.count(tok.special_id) == 1
    assert tok.decode(ids) == f"A.{SPECIAL}B."


def test_trained_tokenizer_knows_the_story_words(tok):
    assert tok.vocab_size == 4096
    for word in [" the", " Lily", " happy", "Once", " upon"]:
        assert len(tok.encode(word)) == 1, word


def test_tokenizer_cases_fixture_is_current(tok):
    # The Rust tokenizer is tested against this file, so it must match this implementation.
    cases = json.loads((ROOT / "fixtures/tokenizer_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        assert tok.encode(case["text"]) == case["ids"], case["text"]
