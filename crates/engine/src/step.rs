//! One token through the model, with the parts of the trace a visualization draws.
//!
//! The full [`Trace`] keeps every tensor; for a single new token most of it is small, and the
//! browser needs only a few views of it, flattened into plain arrays it can index directly.

use crate::error::{Result, bail};
use crate::model::{Model, Session};
use crate::sample::softmax_top_k;
use crate::trace::Trace;

/// What happened inside the model while it processed one token at position `pos`.
#[derive(Debug, Clone)]
pub struct StepTrace {
    pub pos: usize,
    /// `[L, H, pos+1]`: how much each head at each layer attended to every position so far.
    pub attention: Vec<f32>,
    /// `[L, F]`: SwiGLU activations, the "neurons", per layer.
    pub mlp: Vec<f32>,
    /// `[L+1, D]`: the residual stream after the embedding and after each layer.
    pub resid: Vec<f32>,
    /// `[V]`: next-token logits.
    pub logits: Vec<f32>,
    /// Logit lens, when asked for: for the embedding and each layer, the `LENS_K` tokens the model
    /// would predict if it stopped there, `[L+1, LENS_K]`, with their probabilities.
    pub lens_ids: Vec<u32>,
    pub lens_probs: Vec<f32>,
}

pub const LENS_K: usize = 5;

/// Feed one token through `session` and collect the views of its trace; with `lens`, also run
/// the logit lens at every layer (costs about half a forward pass).
pub fn step(model: &Model, session: &mut Session, token: u32, lens: bool) -> Result<StepTrace> {
    let c = &model.config;
    let pos = session.pos();
    let mut trace = Trace::default();
    let logits = session.run(model, &[token], Some(&mut trace))?;

    let get = |name: &str| -> Result<&[f32]> {
        match trace.get(name) {
            Some(t) => Ok(&t.data),
            None => bail!("step: trace has no {name}"),
        }
    };
    let mut attention = Vec::with_capacity(c.n_layers * c.n_heads * (pos + 1));
    let mut mlp = Vec::with_capacity(c.n_layers * c.hidden_dim);
    let mut resid = Vec::with_capacity((c.n_layers + 1) * c.dim);
    resid.extend_from_slice(get("embed")?);
    for l in 0..c.n_layers {
        // attn_weights is [H, 1, pos+1] for a single new token: already the rows we want.
        attention.extend_from_slice(get(&format!("layers.{l}.attn_weights"))?);
        mlp.extend_from_slice(get(&format!("layers.{l}.mlp_act"))?);
        resid.extend_from_slice(get(&format!("layers.{l}.resid_out"))?);
    }
    let (mut lens_ids, mut lens_probs) = (Vec::new(), Vec::new());
    if lens {
        for row in resid.chunks_exact(c.dim) {
            let (ids, probs) = softmax_top_k(&model.lens(row), LENS_K);
            lens_ids.extend(ids);
            lens_probs.extend(probs);
        }
    }
    Ok(StepTrace {
        pos,
        attention,
        mlp,
        resid,
        logits,
        lens_ids,
        lens_probs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_views_have_the_right_shapes() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/tiny/model.safetensors"
        );
        let model = Model::from_safetensors(&std::fs::read(path).unwrap()).unwrap();
        let c = model.config.clone();
        let mut session = Session::new(&model);
        for (i, tok) in [5u32, 6, 7].into_iter().enumerate() {
            let s = step(&model, &mut session, tok, true).unwrap();
            assert_eq!(s.pos, i);
            assert_eq!(s.attention.len(), c.n_layers * c.n_heads * (i + 1));
            assert_eq!(s.mlp.len(), c.n_layers * c.hidden_dim);
            assert_eq!(s.resid.len(), (c.n_layers + 1) * c.dim);
            assert_eq!(s.logits.len(), c.vocab_size);
            // The lens at the last layer is exactly the model's own prediction.
            let last = &s.lens_ids[c.n_layers * LENS_K..];
            assert_eq!(last, softmax_top_k(&s.logits, LENS_K).0.as_slice());
            for row in s.attention.chunks_exact(i + 1) {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "attention row sums to {sum}");
            }
        }
    }
}
