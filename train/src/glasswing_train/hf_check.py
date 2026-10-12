"""Reference data for running Hugging Face models (SmolLM2) on the glasswing engine.

    uv run --with tokenizers --with transformers python -m glasswing_train.hf_check

Needs data/smollm2/ (config.json, model.safetensors, tokenizer.json from SmolLM2-135M-Instruct).
Writes:
- fixtures/smollm2/tokenizer.json + cases.json: ids from Hugging Face's `tokenizers` library
  for awkward strings; the Rust HfTokenizer must reproduce them.
- fixtures/tiny-hf/: a tiny random model in Hugging Face layout (config.json, bf16 weights) and
  the logits PyTorch computes from it, so CI tests `Model::from_hf` without the real weights.
And checks, without writing anything, that our PyTorch model with SmolLM2's weights gives the
same logits as `transformers`' LlamaForCausalLM: the architecture really is the same.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glasswing_train.fixtures import random_tiny_model
from glasswing_train.model import ModelConfig, Transformer

ROOT = Path(__file__).resolve().parents[3]
SMOL = ROOT / "data/smollm2"
OUT = ROOT / "fixtures"

CASES = [
    "",
    "Hello, world!",
    "The year 2026 had 365 days; π ≈ 3.14159.",
    "  two leading spaces, and trailing  ",
    "I'm sure they'll say it's John's, we've seen it, he'd know.",
    "Tabs\tand\r\nnewlines\n\n\nthree",
    "café naïve Zoë Ærøskøbing",
    "Привет, мир! مرحبا بالعالم ١٢٣ שלום",
    "日本語のテキストと漢字",
    "emoji 🐶🚀 and ZWJ 👩‍💻",
    "superscript² roman Ⅻ fraction ½",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "text<|im_end|>more<|endoftext|>",
    "def f(x):\n    return x ** 2  # square",
    "control\x04char\x00s",  # \x04 has no token: tokenizers drops it
]

ARCH = dict(  # SmolLM2-135M, from its config.json
    vocab_size=49152,
    dim=576,
    n_layers=30,
    n_heads=9,
    n_kv_heads=3,
    hidden_dim=1536,
    rope_theta=100000.0,
    norm_eps=1e-5,
)


def tokenizer_cases() -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(SMOL / "tokenizer.json"))
    dst = OUT / "smollm2"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(SMOL / "tokenizer.json", dst / "tokenizer.json")
    rev = (SMOL / "REVISION").read_text().strip()
    (dst / "NOTICE").write_text(
        "tokenizer.json is from HuggingFaceTB/SmolLM2-135M-Instruct (Apache-2.0),\n"
        f"revision {rev}. Used as a test fixture.\n"
    )
    cases = []
    for t in CASES:
        ids = tok.encode(t).ids
        cases.append({"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    lines = ",\n ".join(json.dumps(c, ensure_ascii=False) for c in cases)
    (dst / "cases.json").write_text(f"[\n {lines}\n]\n", encoding="utf-8")
    print(f"tokenizer: {len(cases)} cases")


def load_smollm2(max_seq_len: int = 512) -> Transformer:
    model = Transformer(ModelConfig(**ARCH, max_seq_len=max_seq_len))
    with safe_open(str(SMOL / "model.safetensors"), framework="pt") as f:
        state = {k: f.get_tensor(k).float() for k in f.keys()}
    model.load_state_dict(state, strict=True)  # same parameter names as Hugging Face Llama
    return model.eval()


def check_against_transformers() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMOL)
    hf = AutoModelForCausalLM.from_pretrained(SMOL, torch_dtype=torch.float32).eval()
    ours = load_smollm2()
    ids = torch.tensor([tok.encode("The capital of France is Paris. The capital of Germany is")])
    with torch.no_grad():
        a = hf(ids).logits
        b = ours(ids)
    diff = (a - b).abs().max().item()
    print(
        f"SmolLM2: ours vs transformers, max |logit diff| {diff:.2e}, next token "
        f"{tok.decode(int(b[0, -1].argmax()))!r}"
    )
    assert diff < 1e-3, diff


def tiny_hf_fixture() -> None:
    """Our tiny fixture model in Hugging Face layout, weights rounded to bf16."""
    model = random_tiny_model(seed=3)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).float())  # what the Rust side will load
    c = model.cfg
    dst = OUT / "tiny-hf"
    dst.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": c.dim,
        "intermediate_size": c.hidden_dim,
        "num_hidden_layers": c.n_layers,
        "num_attention_heads": c.n_heads,
        "num_key_value_heads": c.n_kv_heads,
        "vocab_size": c.vocab_size,
        "max_position_embeddings": c.max_seq_len,
        "rope_theta": c.rope_theta,
        "rms_norm_eps": c.norm_eps,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_scaling": None,
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
        str(dst / "model.safetensors"),
    )
    tokens = torch.tensor([[4095, 428, 437, 258, 396, 44, 316, 73]])
    with torch.no_grad():
        logits = model(tokens)[0]
    save_file(
        {"tokens": tokens[0], "logits": logits.contiguous()}, str(dst / "expected.safetensors")
    )
    print(f"tiny-hf: {(dst / 'model.safetensors').stat().st_size / 1e6:.1f} MB")


def main() -> None:
    tokenizer_cases()
    tiny_hf_fixture()
    check_against_transformers()


if __name__ == "__main__":
    main()
