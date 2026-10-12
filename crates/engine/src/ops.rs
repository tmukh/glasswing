//! The handful of operations a Llama-style transformer is made of. All tensors are flat `f32`
//! slices in row-major order: element (r, c) of an [R, C] matrix is at `r * C + c`.

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `x [rows, in] · wᵀ` with `w [out, in]` (PyTorch's `nn.Linear` layout) -> `[rows, out]`.
///
/// Each output is the dot product of an input row with a weight row; both are contiguous in
/// memory, so the inner loop streams through memory in order.
pub fn linear(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0.0; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for (o, out_v) in out[r * out_dim..(r + 1) * out_dim].iter_mut().enumerate() {
            *out_v = dot(xr, &w[o * in_dim..(o + 1) * in_dim]);
        }
    }
    out
}

/// RMSNorm per row: `x / sqrt(mean(x²) + eps) * weight`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let mut out = vec![0.0; x.len()];
    for (xr, or) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean_sq = dot(xr, xr) / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for ((o, &v), &w) in or.iter_mut().zip(xr).zip(weight) {
            *o = v * inv_rms * w;
        }
    }
    out
}

/// Softmax in place, numerically stable: subtracting the max changes nothing mathematically
/// (the factor e^-max cancels) but keeps every exponent <= 0, so nothing overflows to infinity.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU (a.k.a. swish): `x * sigmoid(x)`. Smooth, lets small negatives through.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE tables `[max_seq_len, head_dim / 2]`: angle for position p and pair i is
/// `p * theta^(-2i / head_dim)`.
pub fn rope_tables(max_seq_len: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(max_seq_len * half);
    let mut sin = Vec::with_capacity(max_seq_len * half);
    for p in 0..max_seq_len {
        for &f in &inv_freq {
            let angle = p as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// Rotate each head's dimension pairs (i, i + half) in place.
/// `x` is `[T, n_heads * head_dim]`; position t uses row t of the tables.
pub fn apply_rope(x: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for (t, row) in x.chunks_exact_mut(n_heads * head_dim).enumerate() {
        let (c, s) = (
            &cos[t * half..(t + 1) * half],
            &sin[t * half..(t + 1) * half],
        );
        for head in row.chunks_exact_mut(head_dim) {
            let (x1, x2) = head.split_at_mut(half);
            for i in 0..half {
                let (a, b) = (x1[i], x2[i]);
                x1[i] = a * c[i] - b * s[i];
                x2[i] = b * c[i] + a * s[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_uses_out_in_layout() {
        // w = [[1, 2], [3, 4], [5, 6]] maps 2 inputs to 3 outputs.
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(linear(&[1.0, 1.0], &w, 1, 2, 3), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn softmax_survives_huge_logits() {
        // Without subtracting the max, e^1000 is inf and the result is NaN.
        let mut x = [1000.0, 1000.0, 0.0];
        softmax(&mut x);
        assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6 && x[2] == 0.0);
    }

    #[test]
    fn rmsnorm_gives_unit_rms() {
        let out = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        let rms = (dot(&out, &out) / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let (cos, sin) = rope_tables(4, 4, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 1, 4, &cos, &sin);
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
