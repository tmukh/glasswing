//! The forward pass. Mirrors `train/src/glasswing_train/model.py` step by step; the golden
//! fixtures check that it computes the same numbers.
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

    /// Run the whole sequence and return logits `[T, V]`, row t = scores for the token after t.
    /// With `trace`, every intermediate tensor is recorded under the same names as the Python
    /// reference.
    pub fn forward(&self, tokens: &[u32], mut trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        let c = &self.config;
        let (t_len, d) = (tokens.len(), c.dim);
        if t_len == 0 || t_len > c.max_seq_len {
            bail!("forward: {t_len} tokens, must be 1..={}", c.max_seq_len);
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= c.vocab_size) {
            bail!("forward: token id {bad} >= vocab size {}", c.vocab_size);
        }

        // Embedding: copy each token's row of the table into the residual stream.
        let mut x = Vec::with_capacity(t_len * d);
        for &tok in tokens {
            let t = tok as usize;
            x.extend_from_slice(&self.embed[t * d..(t + 1) * d]);
        }
        record(&mut trace, "embed", &[t_len, d], &x);

        for (i, layer) in self.layers.iter().enumerate() {
            x = self.block(i, layer, x, t_len, &mut trace);
        }

        let x = rmsnorm(&x, &self.norm, c.norm_eps);
        record(&mut trace, "final_norm", &[t_len, d], &x);
        // Tied embeddings: logit for token v = final stream · embedding row v.
        let logits = linear(&x, &self.embed, t_len, d, c.vocab_size);
        record(&mut trace, "logits", &[t_len, c.vocab_size], &logits);
        Ok(logits)
    }

    fn block(
        &self,
        i: usize,
        l: &Layer,
        mut x: Vec<f32>,
        t_len: usize,
        trace: &mut Option<&mut Trace>,
    ) -> Vec<f32> {
        let c = &self.config;
        let (d, hd, f) = (c.dim, c.head_dim(), c.hidden_dim);
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let name = |s: &str| format!("layers.{i}.{s}");

        // --- Attention: read a normalised copy of the stream, add the result back. ---
        let h = rmsnorm(&x, &l.attn_norm, c.norm_eps);
        record(trace, &name("attn_in"), &[t_len, d], &h);

        let mut q = linear(&h, &l.wq, t_len, d, nh * hd); // [T, H*Hd]
        let mut k = linear(&h, &l.wk, t_len, d, nkv * hd); // [T, K*Hd]
        let v = linear(&h, &l.wv, t_len, d, nkv * hd); // [T, K*Hd]
        apply_rope(&mut q, nh, hd, &self.rope_cos, &self.rope_sin);
        apply_rope(&mut k, nkv, hd, &self.rope_cos, &self.rope_sin);
        if trace.is_some() {
            // The reference traces per head: [heads, T, Hd].
            record(
                trace,
                &name("q"),
                &[nh, t_len, hd],
                &heads_first(&q, t_len, nh, hd),
            );
            record(
                trace,
                &name("k"),
                &[nkv, t_len, hd],
                &heads_first(&k, t_len, nkv, hd),
            );
            record(
                trace,
                &name("v"),
                &[nkv, t_len, hd],
                &heads_first(&v, t_len, nkv, hd),
            );
        }

        let (attn, weights) = attention(&q, &k, &v, t_len, nh, nkv, hd);
        if trace.is_some() {
            record(trace, &name("attn_weights"), &[nh, t_len, t_len], &weights);
        }
        let attn_out = linear(&attn, &l.wo, t_len, nh * hd, d);
        record(trace, &name("attn_out"), &[t_len, d], &attn_out);
        add(&mut x, &attn_out);
        record(trace, &name("resid_mid"), &[t_len, d], &x);

        // --- MLP (SwiGLU): per position, D -> F neurons -> D. ---
        let h = rmsnorm(&x, &l.mlp_norm, c.norm_eps);
        record(trace, &name("mlp_in"), &[t_len, d], &h);
        let gate = linear(&h, &l.w_gate, t_len, d, f);
        let up = linear(&h, &l.w_up, t_len, d, f);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        record(trace, &name("mlp_act"), &[t_len, f], &act);
        let mlp_out = linear(&act, &l.w_down, t_len, f, d);
        record(trace, &name("mlp_out"), &[t_len, d], &mlp_out);
        add(&mut x, &mlp_out);
        record(trace, &name("resid_out"), &[t_len, d], &x);
        x
    }
}

/// Causal multi-head attention with grouped key/value heads.
///
/// For each query head and position t: score every position s <= t, softmax the scores, and
/// average the values with those weights. The causal mask is simply that s never goes past t.
/// Returns the output `[T, H*Hd]` and the weights `[H, T, T]` (zero above the diagonal).
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; t_len * nh * hd];
    let mut weights = vec![0.0; nh * t_len * t_len];
    for h in 0..nh {
        let kv = h / group; // query heads 0..group share kv head 0, and so on
        for t in 0..t_len {
            let qt = &q[(t * nh + h) * hd..][..hd];
            let w = &mut weights[(h * t_len + t) * t_len..][..t_len];
            for (s, ws) in w[..=t].iter_mut().enumerate() {
                *ws = dot(qt, &k[(s * nkv + kv) * hd..][..hd]) * scale;
            }
            softmax(&mut w[..=t]);
            let ot = &mut out[(t * nh + h) * hd..][..hd];
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

/// `[T, heads*Hd]` -> `[heads, T, Hd]`.
fn heads_first(x: &[f32], t_len: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for h in 0..heads {
        for t in 0..t_len {
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
