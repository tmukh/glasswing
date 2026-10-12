//! Thin wasm-bindgen layer: the browser calls these, they call the engine.
//! Keep logic out of here so it stays testable natively.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    glasswing_engine::version().to_string()
}
