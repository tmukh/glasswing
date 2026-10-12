//! Hugging Face byte-level BPE tokenizers (GPT-2 family, e.g. SmolLM2), read from `tokenizer.json`.
//!
//! The same algorithm as our own tokenizer, with three differences:
//! 1. **Added tokens** (`<|im_start|>`, ...) are cut out of the text first and map to one id each.
//! 2. **Pre-tokenization** is the GPT-2 regex with Unicode classes (letters `\p{L}`, numbers `\p{N}`),
//!    after splitting every digit into its own piece (SmolLM2's `Digits` pre-tokenizer).
//! 3. **Byte-level alphabet**: GPT-2 stores tokens as strings in which every byte is replaced by a
//!    printable Unicode character (`bytes_to_unicode`), so `Ġ` in the vocabulary means a space.
//!    We map those strings back to real bytes once, at load time, and work on bytes after that.
//!
//! `fixtures/smollm2/cases.json` (from Hugging Face's own `tokenizers` library) checks it.

use std::collections::HashMap;

use fancy_regex::Regex;
use serde::Deserialize;

use crate::error::{Error, Result, bail};

/// GPT-2's split pattern, as used by `tokenizers`' ByteLevel pre-tokenizer.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[derive(Deserialize)]
struct File {
    added_tokens: Vec<AddedToken>,
    pre_tokenizer: serde_json::Value,
    model: BpeModel,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct BpeModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<Merge>,
}

/// Merges are `"a b"` in older files and `["a", "b"]` in newer ones.
#[derive(Deserialize)]
#[serde(untagged)]
enum Merge {
    Joined(String),
    Pair(String, String),
}

pub struct HfTokenizer {
    /// id -> bytes (added tokens: their text).
    vocab: Vec<Vec<u8>>,
    /// (left, right) -> (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Token for each byte; `None` for bytes the vocabulary lacks (some control characters).
    byte_ids: [Option<u32>; 256],
    /// Added tokens, longest first, so `<|im_start|>` wins over any shorter prefix.
    added: Vec<(String, u32)>,
    is_added: Vec<bool>,
    split_digits: bool,
    pattern: Regex,
}

/// GPT-2's byte -> printable character table.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut extra = 0;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    table
}

impl HfTokenizer {
    pub fn from_json(json: &str) -> Result<Self> {
        let file: File =
            serde_json::from_str(json).map_err(|e| Error(format!("tokenizer.json: {e}")))?;
        if file.model.kind != "BPE" {
            bail!("tokenizer.json: model type {} is not BPE", file.model.kind);
        }
        let pre = file.pre_tokenizer.to_string();
        if !pre.contains("ByteLevel") {
            bail!("tokenizer.json: only byte-level pre-tokenizers are supported");
        }
        let split_digits = pre.contains(r#""individual_digits":true"#);

        let table = bytes_to_unicode();
        let mut inverse = HashMap::new();
        for (b, &c) in table.iter().enumerate() {
            inverse.insert(c, b as u8);
        }
        let size = file
            .model
            .vocab
            .values()
            .chain(file.added_tokens.iter().map(|a| &a.id))
            .max()
            .map_or(0, |&m| m as usize + 1);
        let mut vocab = vec![Vec::new(); size];
        let mut is_added = vec![false; size];
        for (s, &id) in &file.model.vocab {
            let bytes: Option<Vec<u8>> = s.chars().map(|c| inverse.get(&c).copied()).collect();
            vocab[id as usize] = bytes
                .ok_or_else(|| Error(format!("tokenizer.json: token {s:?} is not byte-level")))?;
        }
        let mut added = Vec::new();
        for a in &file.added_tokens {
            vocab[a.id as usize] = a.content.as_bytes().to_vec();
            is_added[a.id as usize] = true;
            added.push((a.content.clone(), a.id));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut byte_ids = [None; 256];
        for (b, &c) in table.iter().enumerate() {
            byte_ids[b] = file.model.vocab.get(&c.to_string()).copied();
        }

        let lookup = |s: &str| -> Result<u32> {
            file.model
                .vocab
                .get(s)
                .copied()
                .ok_or_else(|| Error(format!("tokenizer.json: merge part {s:?} not in vocab")))
        };
        let mut merges = HashMap::with_capacity(file.model.merges.len());
        for (rank, m) in file.model.merges.iter().enumerate() {
            let (a, b) = match m {
                Merge::Joined(s) => s
                    .split_once(' ')
                    .ok_or_else(|| Error(format!("tokenizer.json: bad merge {s:?}")))?,
                Merge::Pair(a, b) => (a.as_str(), b.as_str()),
            };
            merges.insert(
                (lookup(a)?, lookup(b)?),
                (rank as u32, lookup(&format!("{a}{b}"))?),
            );
        }

        let pattern = Regex::new(GPT2_PATTERN).map_err(|e| Error(format!("regex: {e}")))?;
        Ok(HfTokenizer {
            vocab,
            merges,
            byte_ids,
            added,
            is_added,
            split_digits,
            pattern,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Id of an added token such as `<|im_end|>`.
    pub fn token_id(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(c, _)| c == content)
            .map(|&(_, id)| id)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest added token in the rest of the text (longest one on ties).
            let next = self
                .added
                .iter()
                .filter_map(|(c, id)| rest.find(c.as_str()).map(|at| (at, c.len(), *id)))
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            match next {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut pieces = Vec::new();
        if self.split_digits {
            // Every numeric character becomes its own piece; the text between stays whole.
            let mut start = 0;
            for (i, c) in text.char_indices() {
                if c.is_numeric() {
                    if start < i {
                        pieces.push(&text[start..i]);
                    }
                    pieces.push(&text[i..i + c.len_utf8()]);
                    start = i + c.len_utf8();
                }
            }
            if start < text.len() {
                pieces.push(&text[start..]);
            }
        } else {
            pieces.push(text);
        }
        for piece in pieces {
            for m in self.pattern.find_iter(piece).flatten() {
                self.bpe(m.as_str().as_bytes(), out);
            }
        }
    }

    fn bpe(&self, chunk: &[u8], out: &mut Vec<u32>) {
        // Bytes without a token are dropped, as `tokenizers` does when there is no unknown token.
        let mut ids: Vec<u32> = chunk
            .iter()
            .filter_map(|&b| self.byte_ids[b as usize])
            .collect();
        loop {
            // The adjacent pair that was learned earliest is merged first.
            let best = ids
                .windows(2)
                .filter_map(|w| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, new)| (rank, w[0], w[1], new))
                })
                .min();
            let Some((_, a, b, new)) = best else { break };
            let pair = (a, b);
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                    merged.push(new);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        out.extend(ids);
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.vocab[id as usize]
    }

    pub fn is_added(&self, id: u32) -> bool {
        self.is_added.get(id as usize).copied().unwrap_or(false)
    }

    /// Text for ids; added tokens are left out when `skip_added` is set.
    pub fn decode(&self, ids: &[u32], skip_added: bool) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&i| !(skip_added && self.is_added(i)))
            .flat_map(|&i| self.token_bytes(i))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
