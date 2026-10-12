//! A safetensors reader, written out instead of pulled in.
//!
//! File layout:
//! ```text
//! [8 bytes: header length N, little-endian u64][N bytes: JSON header][raw tensor bytes]
//! ```
//! The header maps each tensor name to `{"dtype", "shape", "data_offsets": [start, end]}`, with
//! offsets relative to the start of the raw bytes, plus an optional `"__metadata__"` string map.
//! Nothing in the file is executed on load (unlike PyTorch's pickle-based `.pt`).

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    I64,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I64 => 8,
        }
    }
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A parsed file borrowing its bytes; tensors are copied out on request.
pub struct SafeTensors<'a> {
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("safetensors: file too short ({} bytes)", bytes.len());
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let Some(header) = bytes.get(8..8usize.saturating_add(n)) else {
            bail!(
                "safetensors: header length {n} exceeds file size {}",
                bytes.len()
            );
        };
        let data = &bytes[8 + n..];

        let mut json: HashMap<String, serde_json::Value> = serde_json::from_slice(header)
            .map_err(|e| crate::error::Error(format!("safetensors: bad header JSON: {e}")))?;
        let metadata = match json.remove("__metadata__") {
            Some(m) => serde_json::from_value(m)
                .map_err(|e| crate::error::Error(format!("safetensors: bad metadata: {e}")))?,
            None => HashMap::new(),
        };

        let mut entries = HashMap::new();
        for (name, value) in json {
            let raw: RawEntry = serde_json::from_value(value)
                .map_err(|e| crate::error::Error(format!("safetensors: {name}: {e}")))?;
            let dtype = match raw.dtype.as_str() {
                "F32" => Dtype::F32,
                "I64" => Dtype::I64,
                other => bail!("safetensors: {name}: unsupported dtype {other}"),
            };
            let [start, end] = raw.data_offsets;
            let numel: usize = raw.shape.iter().product();
            // Check the header against the data, so a corrupt or hostile file fails here
            // instead of reading out of bounds later.
            if start > end || end > data.len() || end - start != numel * dtype.size() {
                bail!(
                    "safetensors: {name}: offsets {start}..{end} don't fit shape {:?} {} \
                     in {} data bytes",
                    raw.shape,
                    raw.dtype,
                    data.len()
                );
            }
            entries.insert(
                name,
                Entry {
                    dtype,
                    shape: raw.shape,
                    start,
                    end,
                },
            );
        }
        Ok(SafeTensors {
            entries,
            metadata,
            data,
        })
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        match self.entries.get(name) {
            Some(e) => Ok(e),
            None => bail!("safetensors: missing tensor {name}"),
        }
    }

    /// The tensor as f32 values in row-major order, with its shape.
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let e = self.entry(name)?;
        if e.dtype != Dtype::F32 {
            bail!("safetensors: {name} is {:?}, expected F32", e.dtype);
        }
        let values = self.data[e.start..e.end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((e.shape.clone(), values))
    }

    pub fn i64(&self, name: &str) -> Result<(Vec<usize>, Vec<i64>)> {
        let e = self.entry(name)?;
        if e.dtype != Dtype::I64 {
            bail!("safetensors: {name} is {:?}, expected I64", e.dtype);
        }
        let values = self.data[e.start..e.end]
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((e.shape.clone(), values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_f32_and_metadata() {
        let data: Vec<u8> = [1.0f32, -2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = file(
            r#"{"__metadata__":{"k":"v"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &data,
        );
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.metadata()["k"], "v");
        assert_eq!(st.f32("w").unwrap(), (vec![2], vec![1.0, -2.5]));
    }

    #[test]
    fn rejects_offsets_that_do_not_match_the_shape() {
        let bytes = file(
            r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            &[0; 8],
        );
        assert!(SafeTensors::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_offsets_past_the_end() {
        let bytes = file(
            r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
            &[0; 8],
        );
        assert!(SafeTensors::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_header_longer_than_file() {
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(SafeTensors::parse(&bytes).is_err());
    }
}
