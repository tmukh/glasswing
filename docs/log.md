# Log

Short notes on what I did each day and what I got stuck on. Break-it labs get two lines: the symptom,
and the command that exposed it.

## 2026-10-13

Created the repo. Wrote the plan and ADR 0001 (TinyStories, Llama-style ~15M, Rust engine, safetensors).
M0 scaffold: Cargo workspace, uv training project, CI.

M0 shipped: public repo github.com/tmukh/glasswing, CI green (rust: fmt, clippy, test, wasm32 build;
python: ruff, pytest), `rust` and `python` required on main. Working mode for this project: Claude
writes the code and explains every concept (see plan.md).

M1: tokenizer trained on 256 MB of TinyStoriesV2-GPT4 in 15 s (3.93 bytes/token, 97.4% of chunks one
token on valid). Reference model; default size corrected from 7.2M to 15.7M parameters (ADR 0001
amended; caught by a test). Fixture test failed in CI at 1e-5 (Linux vs Windows float summation order),
now 1e-4. RoPE tables were built on CPU only; the lab's GPU run exposed it.

Break-it lab M1, causal mask removed: val loss 0.55 instead of 2.96 after 400 steps, generation garbage.
Exposed by `uv run ruff check .` (F841, unused `future`) and `uv run pytest` (3 failures). Fixed with `git revert`.

M2: Rust forward pass and tokenizer. Parity passed on the first run: 78 traced tensors within 1e-4 of
PyTorch, 11 tokenizer cases identical, locally and in CI. Clippy wanted `is_multiple_of`.

Break-it lab M2, layer 1 q_proj loaded transposed (square, so shapes pass): Rust CI red with
`first divergence at p0.layers.1.q`. Exposed by `cargo test --test parity`. Fixed with `git revert`.
