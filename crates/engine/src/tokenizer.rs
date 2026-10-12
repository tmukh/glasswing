//! Byte-level BPE, the Rust side of `train/src/glasswing_train/tokenizer.py`.
//! `fixtures/tokenizer_cases.json` checks both produce the same ids.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Result, bail};

pub const SPECIAL: &str = "<|endoftext|>";
const FORMAT: &str = "glasswing-bpe-v1";

#[derive(Deserialize)]
struct File {
    format: String,
    special_tokens: HashMap<String, u32>,
    merges: Vec<(u32, u32)>,
}

pub struct Tokenizer {
    ranks: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
    special_id: u32,
}

impl Tokenizer {
    pub fn from_json(json: &str) -> Result<Self> {
        let file: File = serde_json::from_str(json)
            .map_err(|e| crate::error::Error(format!("tokenizer: bad JSON: {e}")))?;
        if file.format != FORMAT {
            bail!("tokenizer: unknown format {:?}", file.format);
        }
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut ranks = HashMap::with_capacity(file.merges.len());
        for (rank, &(a, b)) in file.merges.iter().enumerate() {
            if a as usize >= vocab.len() || b as usize >= vocab.len() {
                bail!("tokenizer: merge {rank} refers to a token that doesn't exist yet");
            }
            let merged = [vocab[a as usize].as_slice(), vocab[b as usize].as_slice()].concat();
            vocab.push(merged);
            ranks.insert((a, b), rank as u32);
        }
        let special_id = vocab.len() as u32;
        if file.special_tokens.get(SPECIAL) != Some(&special_id) {
            bail!("tokenizer: expected {SPECIAL} to have id {special_id}");
        }
        Ok(Tokenizer {
            ranks,
            vocab,
            special_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + 1
    }

    pub fn special_id(&self) -> u32 {
        self.special_id
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for (i, piece) in text.split(SPECIAL).enumerate() {
            if i > 0 {
                out.push(self.special_id);
            }
            for chunk in pretokenize(piece.as_bytes()) {
                out.extend(self.encode_chunk(chunk));
            }
        }
        out
    }

    /// Merge the earliest-learned adjacent pair until no learned pair is left.
    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        loop {
            let best = ids
                .windows(2)
                .filter_map(|w| self.ranks.get(&(w[0], w[1])).map(|&r| (r, w[0], w[1])))
                .min();
            let Some((rank, a, b)) = best else { break };
            let new_id = 256 + rank;
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        ids
    }

    /// The bytes of one token. Byte-level BPE tokens can end mid-character, so this is bytes,
    /// not a string.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        if id == self.special_id {
            SPECIAL.as_bytes()
        } else {
            &self.vocab[id as usize]
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Punct,
    Space,
}

fn class(b: u8) -> Class {
    match b {
        b'A'..=b'Z' | b'a'..=b'z' | 0x80..=0xff => Class::Letter,
        b'0'..=b'9' => Class::Digit,
        b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c => Class::Space,
        _ => Class::Punct,
    }
}

/// Hand-written equivalent of the Python regex
/// ` ?L+| ?D+| ?P+|\s+(?!\S)|\s+` (L letters, D digits, P other non-space).
pub fn pretokenize(text: &[u8]) -> Vec<&[u8]> {
    let n = text.len();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;
        let c = class(text[i]);
        if c == Class::Space {
            if text[i] == b' ' && i + 1 < n && class(text[i + 1]) != Class::Space {
                // A single space before a word belongs to the word.
                let word = class(text[i + 1]);
                i += 2;
                while i < n && class(text[i]) == word {
                    i += 1;
                }
            } else {
                let mut j = i;
                while j < n && class(text[j]) == Class::Space {
                    j += 1;
                }
                // `\s+(?!\S)`: if a word follows, leave the run's last character for it
                // (a space joins the word; anything else becomes its own chunk next round).
                i = if j < n && j - i >= 2 { j - 1 } else { j };
            }
        } else {
            i += 1;
            while i < n && class(text[i]) == c {
                i += 1;
            }
        }
        chunks.push(&text[start..i]);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_matches_the_python_examples() {
        let cases: &[(&str, &[&str])] = &[
            ("Once upon a time.", &["Once", " upon", " a", " time", "."]),
            ("a  b", &["a", " ", " b"]),
            ("end.\n\nNext", &["end", ".", "\n", "\n", "Next"]),
            ("I'm 42!", &["I", "'", "m", " 42", "!"]),
            ("café", &["café"]),
            ("trailing   ", &["trailing", "   "]),
        ];
        for (text, expected) in cases {
            let got: Vec<&[u8]> = pretokenize(text.as_bytes());
            let expected: Vec<&[u8]> = expected.iter().map(|s| s.as_bytes()).collect();
            assert_eq!(got, expected, "{text:?}");
        }
    }
}
