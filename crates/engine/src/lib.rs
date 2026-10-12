//! glasswing-engine: a Llama-style transformer forward pass with no ML framework.
//!
//! Everything the model computes (attention weights, MLP activations, the residual stream,
//! logits) can be recorded in a [`Trace`], so the browser can draw it.

mod error;
pub mod model;
pub mod ops;
pub mod safetensors;
pub mod tokenizer;
pub mod trace;

pub use error::{Error, Result};
pub use model::{Config, Model};
pub use tokenizer::Tokenizer;
pub use trace::Trace;

/// Engine version, shown in the browser so a visitor knows which build they are looking at.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo() {
        assert_eq!(version(), "0.1.0");
    }
}
