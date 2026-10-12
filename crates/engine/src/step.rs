//! One token through the model, with the parts of the trace a visualization draws.
//!
//! The full [`Trace`] keeps every tensor; for a single new token most of it is small, and the
//! browser needs only a few views of it, flattened into plain arrays it can index directly.

use crate::error::{Result, bail};
use crate::model::{Model, Session};
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
}

/// Feed one token through `session` and collect the views of its trace.
pub fn step(model: &Model, session: &mut Session, token: u32) -> Result<StepTrace> {
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
    Ok(StepTrace {
        pos,
        attention,
        mlp,
        resid,
        logits,
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
            let s = step(&model, &mut session, tok).unwrap();
            assert_eq!(s.pos, i);
            assert_eq!(s.attention.len(), c.n_layers * c.n_heads * (i + 1));
            assert_eq!(s.mlp.len(), c.n_layers * c.hidden_dim);
            assert_eq!(s.resid.len(), (c.n_layers + 1) * c.dim);
            assert_eq!(s.logits.len(), c.vocab_size);
            for row in s.attention.chunks_exact(i + 1) {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "attention row sums to {sum}");
            }
        }
    }
}
