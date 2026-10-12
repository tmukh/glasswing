//! glasswing-engine: a Llama-style transformer forward pass with no ML framework.
//!
//! Everything the model computes (attention weights, MLP activations, the residual stream,
//! logits) is meant to be observable, so the browser can draw it. The forward pass arrives in M2.

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
