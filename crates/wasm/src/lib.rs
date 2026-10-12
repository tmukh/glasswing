//! Thin wasm-bindgen layer: the browser calls these, they call the engine.
//! Keep logic out of here so it stays testable natively.
//!
//! JavaScript usage (inside a Web Worker, so the page never freezes):
//! ```js
//! const engine = new Engine(weightsBytes, tokenizerJson);
//! engine.reset(seed, 0.8, 0.9);
//! for (const id of engine.encode_prompt("Once upon a time")) engine.step(id);
//! let id = engine.sample();          // then feed it back with engine.step(id), and so on
//! ```

use glasswing_engine::sample::softmax_top_k;
use glasswing_engine::{Model, Sampler, Session, Tokenizer, step};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    glasswing_engine::version().to_string()
}

#[wasm_bindgen]
pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    session: Session,
    sampler: Sampler,
    last_logits: Vec<f32>,
}

/// The result of one `step`: the views of the trace the visualization draws.
#[wasm_bindgen]
pub struct Step {
    token: u32,
    pos: usize,
    attention: Vec<f32>,
    mlp: Vec<f32>,
    resid: Vec<f32>,
    top_ids: Vec<u32>,
    top_probs: Vec<f32>,
}

#[wasm_bindgen]
impl Step {
    pub fn token(&self) -> u32 {
        self.token
    }
    /// Position of this token in the sequence (0 = the leading <|endoftext|>).
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// `[layers, heads, pos+1]` attention weights of this token over all positions so far.
    pub fn attention(&self) -> Vec<f32> {
        self.attention.clone()
    }
    /// `[layers, hidden]` MLP neuron activations.
    pub fn mlp(&self) -> Vec<f32> {
        self.mlp.clone()
    }
    /// `[layers+1, dim]` residual stream after the embedding and after each layer.
    pub fn resid(&self) -> Vec<f32> {
        self.resid.clone()
    }
    /// The 20 most likely next tokens (temperature 1), most likely first.
    pub fn top_ids(&self) -> Vec<u32> {
        self.top_ids.clone()
    }
    pub fn top_probs(&self) -> Vec<f32> {
        self.top_probs.clone()
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8], tokenizer_json: &str) -> Result<Engine, JsError> {
        let model = Model::from_safetensors(weights).map_err(|e| JsError::new(&e.0))?;
        let tokenizer = Tokenizer::from_json(tokenizer_json).map_err(|e| JsError::new(&e.0))?;
        let session = Session::new(&model);
        Ok(Engine {
            model,
            tokenizer,
            session,
            sampler: Sampler::greedy(),
            last_logits: vec![],
        })
    }

    /// Model shape as JSON: layers, heads, dims, vocabulary, context length.
    pub fn config_json(&self) -> String {
        let c = &self.model.config;
        format!(
            r#"{{"vocab_size":{},"dim":{},"n_layers":{},"n_heads":{},"n_kv_heads":{},"hidden_dim":{},"max_seq_len":{}}}"#,
            c.vocab_size, c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.hidden_dim, c.max_seq_len
        )
    }

    /// Start a new story: empty KV cache, new sampler settings.
    pub fn reset(&mut self, seed: u64, temperature: f32, top_p: f32) {
        self.session.reset();
        self.sampler = Sampler::new(temperature, top_p, seed);
        self.last_logits.clear();
    }

    /// `<|endoftext|>` followed by the prompt's tokens, as in training.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.tokenizer.special_id()];
        ids.extend(self.tokenizer.encode(text));
        ids
    }

    pub fn token_text(&self, id: u32) -> String {
        String::from_utf8_lossy(self.tokenizer.token_bytes(id)).into_owned()
    }

    pub fn end_of_text(&self) -> u32 {
        self.tokenizer.special_id()
    }

    pub fn pos(&self) -> usize {
        self.session.pos()
    }

    pub fn max_seq_len(&self) -> usize {
        self.model.config.max_seq_len
    }

    /// Feed one token and return what happened inside the model.
    pub fn step(&mut self, token: u32) -> Result<Step, JsError> {
        let s =
            step::step(&self.model, &mut self.session, token).map_err(|e| JsError::new(&e.0))?;
        let (top_ids, top_probs) = softmax_top_k(&s.logits, 20);
        self.last_logits = s.logits;
        Ok(Step {
            token,
            pos: s.pos,
            attention: s.attention,
            mlp: s.mlp,
            resid: s.resid,
            top_ids,
            top_probs,
        })
    }

    /// Choose the next token from the last step's logits with the current sampler.
    pub fn sample(&mut self) -> Result<u32, JsError> {
        if self.last_logits.is_empty() {
            return Err(JsError::new("sample: call step first"));
        }
        Ok(self.sampler.sample(&self.last_logits))
    }
}
