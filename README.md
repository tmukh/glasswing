# glasswing

A small language model you can watch think.

glasswing is a Llama-style transformer (15.7M parameters) trained on
[TinyStories](https://arxiv.org/abs/2305.07759), running in the browser on an inference engine written
from scratch in Rust and compiled to WebAssembly. While it writes, you see the tokens, the attention
heads, the neurons firing and the next-token probabilities. You can switch parts of it off and see
what changes.

Status: **M3**: trained (val loss 1.182 after 1.31B tokens, 3 h 38 min on one RTX 3060 Ti); KV cache, sampling and a CLI in Rust. Concepts: [docs/concepts](docs/concepts). See [docs/plan.md](docs/plan.md).

## Layout

| Path | What |
|---|---|
| `crates/engine` | The model: forward pass, sampling, activation traces. No ML framework. |
| `crates/wasm` | Thin wasm-bindgen layer for the browser. |
| `train/` | Tokenizer, PyTorch reference model, training, export to safetensors. |
| `docs/` | Plan, ADRs, learning ledger, log. |

## Develop

```sh
cargo test --workspace
cargo build -p glasswing-wasm --target wasm32-unknown-unknown --release

cd train
uv sync
uv run pytest
```

## Licence

MIT. Trained weights carry the dataset's licence terms too; see ADR 0001.
