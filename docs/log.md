# Log

Short notes on what I did each day and what I got stuck on. Break-it labs get two lines: the symptom,
and the command that exposed it.

## 2026-10-13

Created the repo. Wrote the plan and ADR 0001 (TinyStories, Llama-style ~15M, Rust engine, safetensors).
M0 scaffold: Cargo workspace, uv training project, CI.

M0 shipped: public repo github.com/tmukh/glasswing, CI green (rust: fmt, clippy, test, wasm32 build;
python: ruff, pytest), `rust` and `python` required on main. Working mode for this project: Claude
writes the code and explains every concept (see plan.md).
