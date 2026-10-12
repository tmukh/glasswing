//! The models and the generation loop. Runs on a blocking thread (it's pure CPU work) and sends
//! text pieces through a channel, which the HTTP side streams to the client.

use std::time::{Duration, Instant};

use glasswing_engine::hf_tokenizer::HfTokenizer;
use glasswing_engine::{Model, Sampler, Session, Tokenizer};
use serde::Deserialize;
use tokio::sync::mpsc;

pub enum Tok {
    /// SmolLM2: Hugging Face byte-level BPE with chat tokens.
    Hf(Box<HfTokenizer>),
    /// glasswing: our own BPE.
    Own(Tokenizer),
}

pub struct Loaded {
    pub name: &'static str,
    pub model: Model,
    pub tok: Tok,
    /// Token ids that end a generation.
    pub stop: Vec<u32>,
}

impl Loaded {
    pub fn smollm2(dir: &str, context: usize) -> Result<Self, String> {
        let read =
            |f: &str| std::fs::read(format!("{dir}/{f}")).map_err(|e| format!("{dir}/{f}: {e}"));
        let config = String::from_utf8(read("config.json")?).map_err(|e| e.to_string())?;
        let model =
            Model::from_hf(&config, &read("model.safetensors")?, context).map_err(|e| e.0)?;
        let tok = HfTokenizer::from_json(
            &String::from_utf8(read("tokenizer.json")?).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.0)?;
        let stop = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tok.token_id(t))
            .collect();
        Ok(Loaded {
            name: "smollm2",
            model,
            tok: Tok::Hf(Box::new(tok)),
            stop,
        })
    }

    pub fn glasswing(weights: &str, tokenizer: &str) -> Result<Self, String> {
        let model = Model::from_safetensors(
            &std::fs::read(weights).map_err(|e| format!("{weights}: {e}"))?,
        )
        .map_err(|e| e.0)?;
        let tok = Tokenizer::from_json(
            &std::fs::read_to_string(tokenizer).map_err(|e| format!("{tokenizer}: {e}"))?,
        )
        .map_err(|e| e.0)?;
        let stop = vec![tok.special_id()];
        Ok(Loaded {
            name: "glasswing",
            model,
            tok: Tok::Own(tok),
            stop,
        })
    }

    fn bytes(&self, id: u32) -> &[u8] {
        match &self.tok {
            Tok::Hf(t) if t.is_added(id) => &[],
            Tok::Hf(t) => t.token_bytes(id),
            Tok::Own(t) if id == t.special_id() => &[],
            Tok::Own(t) => t.token_bytes(id),
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Build the prompt ids. SmolLM2 gets its ChatML template; glasswing continues a story.
/// Old messages are dropped until the prompt plus `max_new` fits the context.
pub fn prompt_ids(
    m: &Loaded,
    messages: &[Message],
    prompt: &str,
    max_new: usize,
) -> Result<Vec<u32>, String> {
    let budget = m.model.config.max_seq_len.saturating_sub(max_new);
    match &m.tok {
        Tok::Own(t) => {
            let mut ids = vec![t.special_id()];
            ids.extend(t.encode(prompt));
            if ids.len() > budget {
                return Err(format!(
                    "prompt is {} tokens; at most {budget} fit",
                    ids.len()
                ));
            }
            Ok(ids)
        }
        Tok::Hf(t) => {
            let system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
            let mut msgs: Vec<Message> = messages.to_vec();
            if msgs.is_empty() {
                msgs.push(Message {
                    role: "user".into(),
                    content: prompt.into(),
                });
            }
            loop {
                let mut text = format!("<|im_start|>system\n{system}<|im_end|>\n");
                for msg in &msgs {
                    // Strip chat control tokens from user text, so a message can't forge a turn.
                    let content = msg
                        .content
                        .replace("<|im_start|>", "")
                        .replace("<|im_end|>", "");
                    text.push_str(&format!("<|im_start|>{}\n{content}<|im_end|>\n", msg.role));
                }
                text.push_str("<|im_start|>assistant\n");
                let ids = t.encode(&text);
                if ids.len() <= budget {
                    return Ok(ids);
                }
                if msgs.len() <= 1 {
                    return Err(format!(
                        "message is {} tokens; at most {budget} fit",
                        ids.len()
                    ));
                }
                msgs.remove(0);
            }
        }
    }
}

pub enum Piece {
    Text(String),
    Done {
        reason: &'static str,
        tokens: usize,
        ttft_ms: u64,
        tokens_per_s: f64,
    },
    Error(String),
}

pub struct Params {
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub timeout: Duration,
}

/// Generate and send pieces until a stop token, the token limit, the timeout, or the client
/// going away (the channel closes). Returns the finish reason, the token count and TTFT.
pub fn run(
    m: &Loaded,
    ids: Vec<u32>,
    p: &Params,
    started: Instant,
    tx: &mpsc::Sender<Piece>,
) -> (&'static str, usize, Option<Duration>) {
    let mut session = Session::new(&m.model);
    let mut sampler = Sampler::new(p.temperature, p.top_p, p.seed);
    let v = m.model.config.vocab_size;
    let mut logits = match session.run(&m.model, &ids, None) {
        Ok(l) => l[l.len() - v..].to_vec(),
        Err(e) => {
            let _ = tx.blocking_send(Piece::Error(e.0));
            return ("error", 0, None);
        }
    };
    let mut pending: Vec<u8> = Vec::new(); // bytes of a character split across tokens
    let mut ttft = None;
    let mut n = 0;
    let reason = loop {
        if n >= p.max_new {
            break "length";
        }
        if started.elapsed() > p.timeout {
            break "timeout";
        }
        let next = sampler.sample(&logits);
        if m.stop.contains(&next) {
            break "stop";
        }
        n += 1;
        ttft.get_or_insert_with(|| started.elapsed());
        pending.extend_from_slice(m.bytes(next));
        // Send the longest valid UTF-8 prefix; keep an unfinished character for the next token.
        let valid = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_some() => pending.len(), // invalid, not just unfinished
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            if tx.blocking_send(Piece::Text(text)).is_err() {
                break "client_gone";
            }
        }
        if session.pos() >= m.model.config.max_seq_len {
            break "length";
        }
        logits = match session.run(&m.model, &[next], None) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Piece::Error(e.0));
                break "error";
            }
        };
    };
    (reason, n, ttft)
}
