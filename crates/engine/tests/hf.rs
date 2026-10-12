//! Hugging Face compatibility: tokenizer ids from the `tokenizers` library, and a tiny model in
//! Hugging Face layout (config.json, bf16 weights) against PyTorch. Fixtures from
//! `train/src/glasswing_train/hf_check.py`.
#![cfg(feature = "hf")]

use std::path::PathBuf;

use glasswing_engine::Model;
use glasswing_engine::hf_tokenizer::HfTokenizer;
use glasswing_engine::safetensors::SafeTensors;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[test]
fn smollm2_tokenizer_matches_hugging_face() {
    let json = std::fs::read_to_string(fixtures().join("smollm2/tokenizer.json")).unwrap();
    let tok = HfTokenizer::from_json(&json).unwrap();
    assert_eq!(tok.vocab_size(), 49152);
    assert_eq!(tok.token_id("<|im_end|>"), Some(2));

    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        ids: Vec<u32>,
        decoded: String,
    }
    let cases: Vec<Case> = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("smollm2/cases.json")).unwrap(),
    )
    .unwrap();
    for case in &cases {
        assert_eq!(tok.encode(&case.text), case.ids, "{:?}", case.text);
        assert_eq!(tok.decode(&case.ids, false), case.decoded);
    }
}

#[test]
fn hugging_face_layout_bf16_matches_pytorch() {
    let dir = fixtures().join("tiny-hf");
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
    let model = Model::from_hf(&config, &weights, 64).unwrap();

    let expected_bytes = std::fs::read(dir.join("expected.safetensors")).unwrap();
    let expected = SafeTensors::parse(&expected_bytes).unwrap();
    let (_, ids) = expected.i64("tokens").unwrap();
    let (_, want) = expected.f32("logits").unwrap();
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let got = model.forward(&tokens, None).unwrap();
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() - 1e-4 * b.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(worst <= 1e-4, "logits differ by {worst}");
}
