# Learning ledger

Status: **new** (introduced), **used** (used it, not explained back yet), **confident** (explained back correctly, or predicted right twice).
Concepts already confident in homelab-dev's or night-sky-digest's ledger are not repeated here.

| Concept | Introduced in | Status | Last checked | Notes |
|---|---|---|---|---|
| Cargo workspace, crate types (lib, cdylib), wasm32 target | M0 | new | 2026-10-13 | boilerplate, light treatment |
| rust-toolchain.toml as the single toolchain pin (local + CI) | M0 | new | 2026-10-13 | same idea as digest-pinned base images |
| Byte-level BPE: merges, ranks, encode replays training | M1 | new | 2026-10-13 | bpe merge loop written by Claude; train_bpe uses a heap + incremental pair counts |
| Pre-tokenization (chunks, leading space belongs to the word) | M1 | new | 2026-10-13 |  |
| Vocabulary size trade-off (embedding rows, output layer) | M1 | new | 2026-10-13 | GPT-2's 50k would be 19M params at dim 384 |
| Residual stream (x = x + f(norm(x))) | M1 | new | 2026-10-13 |  |
| Self-attention: query, key, value, scaled dot product, softmax | M1 | new | 2026-10-13 |  |
| Causal mask | M1 | new | 2026-10-13 | lab: without it val loss 0.55 vs 2.96, generation garbage |
| Multi-head attention, o_proj | M1 | new | 2026-10-13 |  |
| MLP / SwiGLU gate, neurons | M1 | new | 2026-10-13 |  |
| RMSNorm vs LayerNorm | M1 | new | 2026-10-13 |  |
| RoPE: rotating pairs, score depends on distance | M1 | new | 2026-10-13 | half-split pairing (HF convention) |
| Grouped-query attention (GQA) | M1 | new | 2026-10-13 | used by the fixture model, needed for SmolLM in M7 |
| Tied embeddings, logits | M1 | new | 2026-10-13 |  |
| safetensors vs pickle (no code on load) | M1 | new | 2026-10-13 |  |
| Golden fixtures, stage-by-stage parity | M1 | new | 2026-10-13 |  |
| Float addition isn't associative; tolerances across machines | M1 | new | 2026-10-13 | CI Linux vs Windows: 1.6e-5 |
| Init scale (std 0.02, residual scaled by depth) vs fixture weights | M1 | new | 2026-10-13 |  |
| safetensors layout: u64 header length, JSON header, raw little-endian data | M2 | new | 2026-10-13 | reader validates offsets vs shape vs size |
| Row-major layout, [out, in] Linear weights, contiguous dot products | M2 | new | 2026-10-13 |  |
| Attention as loops: causal mask = loop bound, GQA = h / group | M2 | new | 2026-10-13 |  |
| Numerically stable softmax (subtract the max) | M2 | new | 2026-10-13 | test feeds 1000s |
| Stage-by-stage parity, first-divergence report | M2 | new | 2026-10-13 | lab: transposed q_proj -> layers.1.q |
| Layout bugs on square matrices pass shape checks | M2 | new | 2026-10-13 | lab M2 |
| Next-token training, cross-entropy, perplexity | M3 | new | 2026-10-13 | train loss 2.91 -> 1.17, val 1.85 -> 1.18 |
| AdamW, weight decay (not on norms) | M3 | new | 2026-10-13 |  |
| LR warmup + cosine decay, gradient clipping, gradient accumulation | M3 | new | 2026-10-13 |  |
| bf16 mixed precision (range vs precision; why not fp16) | M3 | new | 2026-10-13 |  |
| Train vs validation loss, overfitting, fixed eval windows | M3 | new | 2026-10-13 | no overfitting over 2.4 epochs |
| Checkpoints with optimizer state, atomic writes, resume | M3 | new | 2026-10-13 |  |
| KV cache: what is cached, why it is valid, memory cost | M3 | new | 2026-10-13 | 94x vs recompute |
| Sampling: greedy, temperature, top-p, seeded RNG | M3 | new | 2026-10-13 |  |
| Decode is memory-bound; tokens/s ~ bandwidth / model size | M3 | new | 2026-10-13 |  |
| Float sums and SIMD: why 8 accumulators vectorise | M3 | new | 2026-10-13 | 94 -> ~295 tokens/s |
| Testing incremental paths separately (lab: RoPE off by one) | M3 | new | 2026-10-13 | parity stayed green, cache test caught it |
