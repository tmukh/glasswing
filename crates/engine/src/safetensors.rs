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
    F16,
    I64,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
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
                "F16" => Dtype::F16,
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

    /// The tensor as f32 values in row-major order, with its shape. F16 tensors are widened:
    /// half the bytes to download, the same f32 math afterwards.
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let e = self.entry(name)?;
        let bytes = &self.data[e.start..e.end];
        let values = match e.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                .collect(),
            Dtype::I64 => bail!("safetensors: {name} is I64, expected a float tensor"),
        };
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

/// IEEE 754 half precision -> single precision, exactly (every f16 value is an f32 value).
///
/// f16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
/// f32: 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits.
/// Normal numbers: re-bias the exponent, shift the mantissa up by 13 bits.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign, // ±0
        0 => {
            // Subnormal: value = man × 2^-24. Shift until the implicit leading 1 appears;
            // every shift lowers the exponent by one.
            let (mut e, mut m) = (127 - 15 + 1, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13), // inf / NaN
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95); // 1/3 rounded to f16
        assert_eq!(f16_to_f32(0x7bff), 65504.0); // largest f16
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24)); // largest subnormal
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

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
