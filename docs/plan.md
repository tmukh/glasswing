# Plan

Spec: CAREER-PLAN.md §3 Project C. This file is the working version.

**Goal:** a small language model, trained by me, running in the browser on an inference engine I wrote
in Rust, with a visualization that shows what happens inside it while it writes, and controls that let
a visitor switch parts of it off and see what changes.

Definition of done for every milestone: merged to main, CI green, deployed where there is something to
deploy (from M4 on), a line in CAREER-PLAN.md's progress log, and the ledger updated.

## How we work on this project (decided 2026-10-13)

Different from the default tutor mode in D:\dev\CLAUDE.md: **Claude writes the code**, including the core
pieces. In return, every milestone opens with an explanation of each new concept (what it is, why it
exists, how it maps to the code) and closes with a walk-through of how the code we wrote works. No
prediction prompts, no mandatory quiz; I can ask for either. Break-it labs stay, Claude drives them and
explains what broke. The ledger stays honest: a concept moves past **new** only when I explain it back.

## Milestones

| M | Deliverable | Gaps it closes | Core piece (explained in depth) |
|---|---|---|---|
| M0 ✅ | Cargo workspace (engine + wasm crates), uv training project, CI (fmt, clippy, tests, wasm build, ruff, pytest), ADR 0001 | Rust project hygiene | — (boilerplate) |
| M1 | Tokenizer: BPE (vocab 4096) trained on TinyStories; PyTorch reference model (RMSNorm, RoPE, SwiGLU); export to safetensors; **golden fixtures**: a tiny random-init model plus its logits and per-layer activations for fixed prompts | tokenization, transformer architecture | the BPE merge loop |
| M2 | Rust forward pass, no ML framework: load safetensors, embeddings → blocks → logits. **Parity test in CI**: Rust matches PyTorch to 1e-4 on the fixtures | transformer internals, numerics | causal self-attention (scores, mask, softmax, weighted sum) |
| M3 | Train the real model (~15M params) on the 3060 Ti; validation loss and sample quality written up. Rust: KV cache, sampling (temperature, top-p), a CLI, tokens/s benchmark | training, inference performance | the KV cache |
| M4 | WebAssembly build in a Web Worker; **trace API** (attention weights, MLP activations, residual norms, logits per token); f16 weights; size and speed budgets | WASM, browser performance | the trace API shape |
| M5 | The visualization on takodana.dev (three.js): tokens, residual stream, attention arcs, neurons firing, logit bars, sampling | portfolio | the attention-arc mapping (weights → geometry) |
| M6 | Interpretability controls: **ablate** a head or neuron, **logit lens**, **activation patching**. Find one circuit (e.g. an induction head or a name-copying head) and write it up | mechanistic interpretability, evaluation | the ablation experiment and the write-up |
| M7 (stretch) | Same engine loads a real small model (SmolLM-135M); native build served on k3s with Prometheus metrics (tokens/s, time to first token) | serving, SRE | — |

## Break-it lab ideas

- **M1:** drop the causal mask during training. Prediction: loss falls suspiciously fast, because the model
  reads the answer from the future. Generation is garbage.
- **M2:** load one weight matrix transposed. The parity test should catch it; which layer fails first?
- **M2/M3:** softmax without subtracting the max. Works on small logits, gives `NaN` on large ones.
- **M3:** RoPE with positions off by one in the KV-cache path only. Cached and uncached generation diverge.
- **M4:** ship f32 weights instead of f16 and watch the page-load budget fail.

## Calendar

Three fixed 2-hour blocks a week (CAREER-PLAN §5). M3 includes one overnight training run.
