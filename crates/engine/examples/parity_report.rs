//! Per-stage parity with PyTorch, as JSON, for the blog (part 4).
//!
//!     cargo run --release -p glasswing-engine --example parity_report -- \
//!         fixtures/tiny/model.safetensors fixtures/tiny/expected.safetensors > report.json
//!
//! For prompt 0 of the golden fixtures: every traced tensor in computation order, with the largest
//! absolute difference between this engine and PyTorch. Run it on a deliberately broken copy of
//! the weights to see where a bug first shows up.

use glasswing_engine::safetensors::SafeTensors;
use glasswing_engine::{Model, Trace};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = Model::from_safetensors(&std::fs::read(&args[1]).unwrap()).unwrap();
    let expected_bytes = std::fs::read(&args[2]).unwrap();
    let expected = SafeTensors::parse(&expected_bytes).unwrap();
    let (_, ids) = expected.i64("p0.tokens").unwrap();
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let mut trace = Trace::default();
    model.forward(&tokens, Some(&mut trace)).unwrap();

    let rows: Vec<String> = trace
        .tensors
        .iter()
        .map(|t| {
            let (_, want) = expected.f32(&format!("p0.{}", t.name)).unwrap();
            let max_diff = t
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_abs = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            format!(
                r#"{{"name":"{}","shape":{:?},"max_diff":{:e},"max_abs":{}}}"#,
                t.name, t.shape, max_diff, max_abs
            )
        })
        .collect();
    println!("[{}]", rows.join(","));
}
