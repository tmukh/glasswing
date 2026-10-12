//! The forward pass. Mirrors `train/src/glasswing_train/model.py` step by step; the golden
//! fixtures check that it computes the same numbers.
//!
//! A [`Session`] holds the KV cache: keys and values of every token processed so far. Feeding it
//! tokens one at a time costs one token's work per step instead of re-running the whole sequence.
//! [`Model::forward`] is just a fresh session given all tokens at once, so the parity test covers
//! the cached path too.
//!
//! Shapes: T = tokens, D = dim, H = query heads, K = key/value heads, Hd = head dim,
//! F = MLP hidden dim, V = vocabulary. Activations are row-major `[T, ...]`: one row per token.

use serde::Deserialize;

use crate::error::{Result, bail};
use crate::ops::{apply_rope, dot, linear, rmsnorm, rope_tables, silu, softmax};
use crate::safetensors::SafeTensors;
use crate::trace::Trace;

pub const FORMAT: &str = "glasswing-model-v1";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

struct Layer {
    attn_norm: Vec<f32>, // [D]
    wq: Vec<f32>,        // [H*Hd, D]
    wk: Vec<f32>,        // [K*Hd, D]
    wv: Vec<f32>,        // [K*Hd, D]
    wo: Vec<f32>,        // [D, H*Hd]
    mlp_norm: Vec<f32>,  // [D]
    w_gate: Vec<f32>,    // [F, D]
    w_up: Vec<f32>,      // [F, D]
    w_down: Vec<f32>,    // [D, F]
}

pub struct Model {
    pub config: Config,
    embed: Vec<f32>, // [V, D], also the output projection (tied)
    layers: Vec<Layer>,
    norm: Vec<f32>, // [D]
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Model {
    /// Load from the bytes of a `.safetensors` file written by `glasswing_train.export`.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::parse(bytes)?;
        if st.metadata().get("format").map(String::as_str) != Some(FORMAT) {
            bail!("model: not a {FORMAT} file");
        }
        let Some(config_json) = st.metadata().get("config") else {
            bail!("model: no config in metadata");
        };
        let config: Config = serde_json::from_str(config_json)
            .map_err(|e| crate::error::Error(format!("model: bad config: {e}")))?;
        let c = &config;
        if !c.dim.is_multiple_of(c.n_heads)
            || !c.n_heads.is_multiple_of(c.n_kv_heads)
            || !c.head_dim().is_multiple_of(2)
        {
            bail!("model: inconsistent head configuration {c:?}");
        }
        let (d, hd, f) = (c.dim, c.head_dim(), c.hidden_dim);

        // Every weight is checked against the shape the config implies.
        let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let (actual, data) = st.f32(name)?;
            if actual != shape {
                bail!("model: {name} has shape {actual:?}, expected {shape:?}");
            }
            Ok(data)
        };

        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let p = format!("model.layers.{i}.");
            layers.push(Layer {
                attn_norm: get(&format!("{p}input_layernorm.weight"), &[d])?,
                wq: get(&format!("{p}self_attn.q_proj.weight"), &[c.n_heads * hd, d])?,
                wk: get(
                    &format!("{p}self_attn.k_proj.weight"),
                    &[c.n_kv_heads * hd, d],
                )?,
                wv: get(
                    &format!("{p}self_attn.v_proj.weight"),
                    &[c.n_kv_heads * hd, d],
                )?,
                wo: get(&format!("{p}self_attn.o_proj.weight"), &[d, c.n_heads * hd])?,
                mlp_norm: get(&format!("{p}post_attention_layernorm.weight"), &[d])?,
                w_gate: get(&format!("{p}mlp.gate_proj.weight"), &[f, d])?,
                w_up: get(&format!("{p}mlp.up_proj.weight"), &[f, d])?,
                w_down: get(&format!("{p}mlp.down_proj.weight"), &[d, f])?,
            });
        }
        let (rope_cos, rope_sin) = rope_tables(c.max_seq_len, hd, c.rope_theta);
        Ok(Model {
            embed: get("model.embed_tokens.weight", &[c.vocab_size, d])?,
            norm: get("model.norm.weight", &[d])?,
            layers,
            rope_cos,
            rope_sin,
            config,
        })
    }

    /// Number of weights (the tied embedding counted once).
    pub fn num_params(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm.len()
                    + l.wq.len()
                    + l.wk.len()
                    + l.wv.len()
                    + l.wo.len()
                    + l.mlp_norm.len()
                    + l.w_gate.len()
                    + l.w_up.len()
                    + l.w_down.len()
            })
            .sum();
        self.embed.len() + self.norm.len() + per_layer
    }

    /// Run a whole sequence from scratch and return logits `[T, V]`, row t = scores for the token
    /// after t. With `trace`, every intermediate tensor is recorded under the same names as the
    /// Python reference.
    pub fn forward(&self, tokens: &[u32], trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        Session::new(self).run(self, tokens, trace)
    }
}

/// One generation in progress: the KV cache for the tokens seen so far. It doesn't hold the
/// model, so an owner (the browser engine) can keep both side by side; pass the same model to
/// every `run`.
pub struct Session {
    /// Per layer: keys and values for positions `0..pos`, row-major `[max_seq_len, K*Hd]`.
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pos: usize,
}

impl Session {
    pub fn new(model: &Model) -> Self {
        let c = &model.config;
        let size = c.max_seq_len * c.n_kv_heads * c.head_dim();
        Session {
            k_cache: vec![vec![0.0; size]; c.n_layers],
            v_cache: vec![vec![0.0; size]; c.n_layers],
            pos: 0,
        }
    }

    /// Number of tokens processed so far (the position the next token will get).
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Process `tokens` as positions `pos..pos+n`, append their keys and values to the cache, and
    /// return logits `[n, V]`. Attention reads the cache, so earlier tokens are never recomputed.
    pub fn run(
        &mut self,
        model: &Model,
        tokens: &[u32],
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>> {
        let m = model;
        let c = &m.config;
        let (n, d) = (tokens.len(), c.dim);
        let cache_len = c.max_seq_len * c.n_kv_heads * c.head_dim();
        if self.k_cache.len() != c.n_layers || self.k_cache[0].len() != cache_len {
            bail!("run: this session was created for a different model");
        }
        if n == 0 || self.pos + n > c.max_seq_len {
            bail!(
                "run: {n} new tokens at position {}, context is {}",
                self.pos,
                c.max_seq_len
            );
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= c.vocab_size) {
            bail!("run: token id {bad} >= vocab size {}", c.vocab_size);
        }

        // Embedding: copy each token's row of the table into the residual stream.
        let mut x = Vec::with_capacity(n * d);
        for &tok in tokens {
            let t = tok as usize;
            x.extend_from_slice(&m.embed[t * d..(t + 1) * d]);
        }
        record(&mut trace, "embed", &[n, d], &x);

        for i in 0..m.layers.len() {
            x = self.block(m, i, x, n, &mut trace);
        }
        self.pos += n;

        let x = rmsnorm(&x, &m.norm, c.norm_eps);
        record(&mut trace, "final_norm", &[n, d], &x);
        // Tied embeddings: logit for token v = final stream · embedding row v.
        let logits = linear(&x, &m.embed, n, d, c.vocab_size);
        record(&mut trace, "logits", &[n, c.vocab_size], &logits);
        Ok(logits)
    }

    fn block(
        &mut self,
        m: &Model,
        i: usize,
        mut x: Vec<f32>,
        n: usize,
        trace: &mut Option<&mut Trace>,
    ) -> Vec<f32> {
        let l = &m.layers[i];
        let c = &m.config;
        let (d, hd, f) = (c.dim, c.head_dim(), c.hidden_dim);
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let (start, end) = (self.pos, self.pos + n);
        let name = |s: &str| format!("layers.{i}.{s}");

        // --- Attention: read a normalised copy of the stream, add the result back. ---
        let h = rmsnorm(&x, &l.attn_norm, c.norm_eps);
        record(trace, &name("attn_in"), &[n, d], &h);

        let mut q = linear(&h, &l.wq, n, d, nh * hd); // [n, H*Hd]
        let mut k = linear(&h, &l.wk, n, d, nkv * hd); // [n, K*Hd]
        let v = linear(&h, &l.wv, n, d, nkv * hd); // [n, K*Hd]
        // Rotate by each token's absolute position, not its index in this call.
        apply_rope(&mut q, nh, hd, &m.rope_cos, &m.rope_sin, start);
        apply_rope(&mut k, nkv, hd, &m.rope_cos, &m.rope_sin, start);
        if trace.is_some() {
            // The reference traces per head: [heads, n, Hd].
            record(trace, &name("q"), &[nh, n, hd], &heads_first(&q, n, nh, hd));
            record(
                trace,
                &name("k"),
                &[nkv, n, hd],
                &heads_first(&k, n, nkv, hd),
            );
            record(
                trace,
                &name("v"),
                &[nkv, n, hd],
                &heads_first(&v, n, nkv, hd),
            );
        }

        // Append the new keys and values to the cache.
        let kv = nkv * hd;
        self.k_cache[i][start * kv..end * kv].copy_from_slice(&k);
        self.v_cache[i][start * kv..end * kv].copy_from_slice(&v);

        let (attn, weights) = attention(
            &q,
            &self.k_cache[i][..end * kv],
            &self.v_cache[i][..end * kv],
            start,
            n,
            nh,
            nkv,
            hd,
        );
        if trace.is_some() {
            record(trace, &name("attn_weights"), &[nh, n, end], &weights);
        }
        let attn_out = linear(&attn, &l.wo, n, nh * hd, d);
        record(trace, &name("attn_out"), &[n, d], &attn_out);
        add(&mut x, &attn_out);
        record(trace, &name("resid_mid"), &[n, d], &x);

        // --- MLP (SwiGLU): per position, D -> F neurons -> D. ---
        let h = rmsnorm(&x, &l.mlp_norm, c.norm_eps);
        record(trace, &name("mlp_in"), &[n, d], &h);
        let gate = linear(&h, &l.w_gate, n, d, f);
        let up = linear(&h, &l.w_up, n, d, f);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        record(trace, &name("mlp_act"), &[n, f], &act);
        let mlp_out = linear(&act, &l.w_down, n, f, d);
        record(trace, &name("mlp_out"), &[n, d], &mlp_out);
        add(&mut x, &mlp_out);
        record(trace, &name("resid_out"), &[n, d], &x);
        x
    }
}

/// Causal multi-head attention with grouped key/value heads, over a KV cache.
///
/// `q` holds `n` new queries at absolute positions `start..start+n`; `k`/`v` hold keys and values
/// for positions `0..start+n`. Query at position t scores every position s <= t, softmaxes the
/// scores and averages the values with those weights. The causal mask is simply that s never goes
/// past t. Returns the output `[n, H*Hd]` and the weights `[H, n, start+n]` (zero after t).
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    start: usize,
    n: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = start + n;
    let group = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; n * nh * hd];
    let mut weights = vec![0.0; nh * n * total];
    for h in 0..nh {
        let kv = h / group; // query heads 0..group share kv head 0, and so on
        for r in 0..n {
            let t = start + r; // absolute position of this query
            let qt = &q[(r * nh + h) * hd..][..hd];
            let w = &mut weights[(h * n + r) * total..][..total];
            for (s, ws) in w[..=t].iter_mut().enumerate() {
                *ws = dot(qt, &k[(s * nkv + kv) * hd..][..hd]) * scale;
            }
            softmax(&mut w[..=t]);
            let ot = &mut out[(r * nh + h) * hd..][..hd];
            for (s, &ws) in w[..=t].iter().enumerate() {
                for (o, &vv) in ot.iter_mut().zip(&v[(s * nkv + kv) * hd..][..hd]) {
                    *o += ws * vv;
                }
            }
        }
    }
    (out, weights)
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// `[n, heads*Hd]` -> `[heads, n, Hd]`.
fn heads_first(x: &[f32], n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for h in 0..heads {
        for t in 0..n {
            out.extend_from_slice(&x[(t * heads + h) * hd..][..hd]);
        }
    }
    out
}

fn record(trace: &mut Option<&mut Trace>, name: &str, shape: &[usize], data: &[f32]) {
    if let Some(tr) = trace.as_deref_mut() {
        tr.record(name, shape, data);
    }
}
