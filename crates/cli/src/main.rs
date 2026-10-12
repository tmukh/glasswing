//! `glasswing`: generate a story, or measure how fast the engine is.
//!
//!     glasswing --model checkpoints/main/model.safetensors --prompt "Once upon a time"
//!     glasswing --model ... --bench
//!     glasswing --model ... --no-cache --bench      # the same, recomputing everything each step

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use glasswing_engine::{Model, Sampler, Session, Tokenizer};

struct Args {
    model: String,
    tokenizer: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    bench: bool,
    no_cache: bool,
}

const USAGE: &str = "usage: glasswing --model FILE [--tokenizer FILE] [--prompt TEXT] \
[--max-tokens N] [--temperature T] [--top-p P] [--seed S] [--bench] [--no-cache]";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        tokenizer: "assets/tokenizer.json".into(),
        prompt: "Once upon a time".into(),
        max_tokens: 200,
        temperature: 0.8,
        top_p: 0.9,
        seed: 0,
        bench: false,
        no_cache: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        let bad = |e: &dyn std::fmt::Display| format!("{flag}: {e}");
        match flag.as_str() {
            "--model" => a.model = value()?,
            "--tokenizer" => a.tokenizer = value()?,
            "--prompt" => a.prompt = value()?,
            "--max-tokens" => a.max_tokens = value()?.parse().map_err(|e| bad(&e))?,
            "--temperature" => a.temperature = value()?.parse().map_err(|e| bad(&e))?,
            "--top-p" => a.top_p = value()?.parse().map_err(|e| bad(&e))?,
            "--seed" => a.seed = value()?.parse().map_err(|e| bad(&e))?,
            "--bench" => a.bench = true,
            "--no-cache" => a.no_cache = true,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.model.is_empty() {
        return Err(USAGE.into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
    let model = Model::from_safetensors(&read(&args.model)?).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_json(
        &String::from_utf8(read(&args.tokenizer)?).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    // Stories start with <|endoftext|>, as in training.
    let mut tokens = vec![tok.special_id()];
    tokens.extend(tok.encode(&args.prompt));
    let mut sampler = if args.bench {
        Sampler::greedy() // fixed output, so runs are comparable
    } else {
        Sampler::new(args.temperature, args.top_p, args.seed)
    };

    let v = model.config.vocab_size;
    let max_new = args
        .max_tokens
        .min(model.config.max_seq_len.saturating_sub(tokens.len()));
    let mut session = Session::new(&model);
    let mut out = std::io::stdout().lock();
    if !args.bench {
        write!(out, "{}", args.prompt).unwrap();
    }

    // Prefill: the whole prompt in one call.
    let prompt_len = tokens.len();
    let t0 = Instant::now();
    let logits = session
        .run(&model, &tokens, None)
        .map_err(|e| e.to_string())?;
    let mut next = sampler.sample(&logits[logits.len() - v..]);
    let prefill = t0.elapsed();

    // Decode: one token per step.
    let t1 = Instant::now();
    let mut generated = 0;
    while generated < max_new && next != tok.special_id() {
        tokens.push(next);
        generated += 1;
        if !args.bench {
            // A token can end mid-character; decoding byte-level tokens one by one is fine
            // for this ASCII-heavy model.
            write!(out, "{}", tok.decode(&[next])).unwrap();
            out.flush().unwrap();
        }
        if tokens.len() >= model.config.max_seq_len {
            break;
        }
        let logits = if args.no_cache {
            // Without the cache: recompute every position for every new token.
            let all = model.forward(&tokens, None).map_err(|e| e.to_string())?;
            all[all.len() - v..].to_vec()
        } else {
            session
                .run(&model, &[next], None)
                .map_err(|e| e.to_string())?
        };
        next = sampler.sample(&logits);
    }
    let decode = t1.elapsed();
    writeln!(out).unwrap();

    if args.bench {
        // Weights are f32 in memory even when the file is f16: 4 bytes read per weight per token.
        let weights_mb = model.num_params() as f64 * 4.0 / 1e6;
        let tps = generated as f64 / decode.as_secs_f64();
        println!(
            "prompt {} tokens in {:.1} ms; generated {generated} tokens in {:.0} ms: \
             {tps:.1} tokens/s ({}), {weights_mb:.0} MB of f32 weights -> {:.1} GB/s of weight reads",
            prompt_len,
            prefill.as_secs_f64() * 1e3,
            decode.as_secs_f64() * 1e3,
            if args.no_cache {
                "no cache"
            } else {
                "KV cache"
            },
            tps * weights_mb / 1e3,
        );
    }
    Ok(())
}
