//! Parity with the PyTorch reference, on the golden fixtures from M1.
//!
//! For each prompt, run the Rust forward pass with tracing and compare every traced tensor in
//! the order it was computed. The first mismatch is reported, because one bug makes everything
//! after it differ too: "p1.layers.1.q" points at the bug, "p1.logits" doesn't.

use std::path::PathBuf;

use glasswing_engine::safetensors::SafeTensors;
use glasswing_engine::{Ablation, Model, Session, Tokenizer, Trace};

// Same bar as the Python fixture test: float sums differ across machines in the last bits.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-4;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn read(path: PathBuf) -> Vec<u8> {
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn forward_matches_pytorch() {
    let model = Model::from_safetensors(&read(fixtures().join("tiny/model.safetensors"))).unwrap();
    let expected_bytes = read(fixtures().join("tiny/expected.safetensors"));
    let expected = SafeTensors::parse(&expected_bytes).unwrap();
    let prompts: Vec<String> = serde_json::from_str(&expected.metadata()["prompts"]).unwrap();
    assert!(!prompts.is_empty());

    for p in 0..prompts.len() {
        let (_, ids) = expected.i64(&format!("p{p}.tokens")).unwrap();
        let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let mut trace = Trace::default();
        model.forward(&tokens, Some(&mut trace)).unwrap();

        let prefix = format!("p{p}.");
        let expected_names: Vec<&str> = expected
            .names()
            .filter_map(|n| n.strip_prefix(&prefix))
            .filter(|n| *n != "tokens")
            .collect();
        for name in &expected_names {
            assert!(
                trace.get(name).is_some(),
                "Rust trace has no {prefix}{name}"
            );
        }

        for t in &trace.tensors {
            let (shape, want) = expected.f32(&format!("{prefix}{}", t.name)).unwrap();
            assert_eq!(t.shape, shape, "{prefix}{}: shape", t.name);
            let (worst, idx) = t
                .data
                .iter()
                .zip(&want)
                .enumerate()
                .map(|(i, (&a, &b))| ((a - b).abs() - RTOL * b.abs(), i))
                .fold(
                    (f32::NEG_INFINITY, 0),
                    |acc, x| if x.0 > acc.0 { x } else { acc },
                );
            assert!(
                worst <= ATOL && t.data.iter().all(|v| v.is_finite()),
                "first divergence at {prefix}{} {:?}: index {idx}: rust {} vs pytorch {}",
                t.name,
                t.shape,
                t.data[idx],
                want[idx]
            );
        }
    }
}

#[test]
fn tokenizer_matches_python() {
    let tok = Tokenizer::from_json(
        &std::fs::read_to_string(fixtures().join("../assets/tokenizer.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(tok.vocab_size(), 4096);

    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        ids: Vec<u32>,
    }
    let cases: Vec<Case> = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("tokenizer_cases.json")).unwrap(),
    )
    .unwrap();
    for case in &cases {
        assert_eq!(tok.encode(&case.text), case.ids, "{:?}", case.text);
        assert_eq!(tok.decode(&case.ids), case.text);
    }
}

#[test]
fn rejects_bad_input() {
    let model = Model::from_safetensors(&read(fixtures().join("tiny/model.safetensors"))).unwrap();
    assert!(model.forward(&[], None).is_err());
    assert!(model.forward(&[5000], None).is_err());
    let too_long = vec![1; model.config.max_seq_len + 1];
    assert!(model.forward(&too_long, None).is_err());
}

#[test]
fn kv_cache_gives_the_same_logits_as_a_full_pass() {
    // Generation feeds one token at a time through the cache; the result must not depend on
    // how the sequence was split up.
    let model = Model::from_safetensors(&read(fixtures().join("tiny/model.safetensors"))).unwrap();
    let tokens: Vec<u32> = (0..30).map(|i| (i * 131 + 7) % 4096).collect();
    let v = model.config.vocab_size;
    let full = model.forward(&tokens, None).unwrap();

    for split in [vec![1; 30], vec![10, 1, 1, 18], vec![29, 1]] {
        let mut session = Session::new(&model);
        let mut at = 0;
        for n in split {
            let logits = session.run(&model, &tokens[at..at + n], None).unwrap();
            for (r, row) in logits.chunks_exact(v).enumerate() {
                let want = &full[(at + r) * v..][..v];
                let diff = row
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(diff < 1e-4, "position {}: max diff {diff}", at + r);
            }
            at += n;
        }
        assert_eq!(session.pos(), 30);
    }
}

#[test]
fn session_refuses_to_run_past_the_context() {
    let model = Model::from_safetensors(&read(fixtures().join("tiny/model.safetensors"))).unwrap();
    let mut session = Session::new(&model);
    let max = model.config.max_seq_len;
    session.run(&model, &vec![1; max], None).unwrap();
    assert!(session.run(&model, &[1], None).is_err());
}

#[test]
fn ablation_changes_only_what_it_should() {
    let model = Model::from_safetensors(&read(fixtures().join("tiny/model.safetensors"))).unwrap();
    let tokens: Vec<u32> = (0..12).map(|i| (i * 97 + 3) % 4096).collect();
    let run = |ablation: Ablation| {
        let mut s = Session::new(&model);
        s.ablation = ablation;
        let mut trace = Trace::default();
        let logits = s.run(&model, &tokens, Some(&mut trace)).unwrap();
        (logits, trace)
    };
    let (base, _) = run(Ablation::default());
    let (head, _) = run(Ablation {
        heads: vec![(1, 2)],
        neurons: vec![],
    });
    assert_ne!(base, head, "ablating a head changes the output");

    let (_, trace) = run(Ablation {
        heads: vec![],
        neurons: vec![(0, 5), (1, 7)],
    });
    let f = model.config.hidden_dim;
    let act0 = &trace.get("layers.0.mlp_act").unwrap().data;
    assert!((0..tokens.len()).all(|r| act0[r * f + 5] == 0.0));
    let act1 = &trace.get("layers.1.mlp_act").unwrap().data;
    assert!((0..tokens.len()).all(|r| act1[r * f + 7] == 0.0));
    assert!(act0.iter().any(|&a| a != 0.0));

    let mut s = Session::new(&model);
    s.ablation = Ablation {
        heads: vec![(9, 0)],
        neurons: vec![],
    };
    assert!(
        s.run(&model, &tokens, None).is_err(),
        "out-of-range ablation is refused"
    );
}
