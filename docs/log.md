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

M3: training ran 00:51–04:35, 40,000 steps, 1.31B tokens, final val loss 1.182 (no overfitting).
Rust greedy output identical to PyTorch on the final weights; native ~280–320 tokens/s.
KV cache 94x faster than recomputing; dot() with 8 accumulators 94 -> ~295 tokens/s.

Break-it lab M3, RoPE position off by one only for single-token decode: fixture parity stayed green,
`cargo test --test parity` failed in kv_cache_gives_the_same_logits_as_a_full_pass (position 10, not 0:
a constant shift is invisible to RoPE). compare.py: one story diverged at char 411. Fixed with `git revert`.

M4: wasm engine (288 KB) in a Web Worker, 285 traced tokens/s in Node/V8 (193 without simd128); f16
weights 31.5 MB, greedy stories unchanged; CI web job (size budget, wasm == native). Release v0.1.0.
Break-it lab M4 skipped (the planned one, shipping f32 weights past the budget, is covered by the
size check in CI and the f16 conversion tests).

M7: SmolLM2-135M-Instruct on the engine. Our PyTorch model with its weights matches transformers to
3e-5; Rust greedy answers identical to transformers. HF tokenizer matches `tokenizers` on 15 cases
(it drops bytes with no token, e.g. 0x04; so do we). Server: SSE, per-IP token bucket, 2 concurrent,
256 tokens, 60 s. Locally 31-36 tokens/s, first token ~450 ms.
Image test in CI found two bugs before any deploy: a one-character container name, and model files
fetched with ADD being mode 600 (unreadable for the non-root user). `ADD --chmod=644`.

M6: induction circuit L3H3/L3H1 (previous-token) -> L4H4 (induction). Circuit off: second copy of
repeated text 1.40 -> 1.59 (random heads 1.42); name task logit diff 10.4 -> 7.8. Patching: the name
moves to the last position in layers 5-7; logit lens 0.00 at layer 4 -> 0.36 at layer 5.
First analysis run had two bugs: random tokens as the ablation test (loss 11.6, the model just
doesn't do random text) and a double space in two templates. Both fixed before the final run.

M7 deployed: glasswing.takodana.dev live (homelab-dev #42), image sha-dffa5be, cosign verified. On the
cluster SmolLM2 runs at 8-9 tokens/s (first token ~2 s), vs 31-36 locally: slower node, 3 threads.
Break-it lab M7, NetworkPolicy without Traefik's namespace (homelab-dev #43, reverted in #44): my
prediction (immediate 502) was wrong at first. The site kept answering 200 for minutes, because a
NetworkPolicy doesn't cut established connections and Traefik reuses keep-alive connections. Only
after they idled out: 502 from `curl https://glasswing.takodana.dev/healthz`; `kubectl -n glasswing
describe networkpolicy glasswing` showed the missing namespace. Pod stayed Ready (kubelet probes
come from the node). Also: one kubectl write (an Argo CD refresh annotation) before I was told
forced syncs are fine; Argo CD force syncs used for the lab after that.

M5 + M6 on the site (takodana.dev #4, homelab-dev #45): https://takodana.dev/projects/glasswing/ and
the write-up /writing/glasswing-induction-heads/. Release files fetched and sha256-verified at build;
Cloudflare caches model.bin (cf-cache-status MISS then HIT). One image build on main failed on a
transient Google Fonts download inside Docker; a re-run passed. Live browser test: same seed, circuit
off -> a different story; no console errors.
Break-it lab M5 skipped (the live checks above replaced it).
