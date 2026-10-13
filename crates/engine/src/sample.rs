//! Choosing the next token from logits: greedy, temperature, top-p, with a seeded RNG so a run
//! can be replayed exactly (tests, and the same story twice in the browser).

use crate::ops::softmax;

/// xorshift64*: tiny, fast, good enough for sampling, and the same on every platform.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Mix the seed so 0, 1, 2... give unrelated streams; the state must never be 0.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1), from the top 24 bits (an f32 mantissa's worth).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    /// 0 = greedy. Below 1 sharpens the distribution, above 1 flattens it.
    pub temperature: f32,
    /// Keep the smallest set of top tokens whose probabilities sum to at least this. 1 = all.
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            rng: Rng::new(seed),
        }
    }

    pub fn greedy() -> Self {
        Sampler::new(0.0, 1.0, 0)
    }

    /// Pick a token id from one position's logits `[V]`.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let mut probs: Vec<f32> = logits.iter().map(|l| l / self.temperature).collect();
        softmax(&mut probs);

        // Candidates by probability, highest first (ties: lower id first, deterministic).
        let mut order: Vec<u32> = (0..probs.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b as usize]
                .total_cmp(&probs[a as usize])
                .then(a.cmp(&b))
        });
        // Nucleus: keep tokens while the mass *before* them is still below top_p, so the token
        // that crosses the threshold is included (same rule as the Python reference).
        let mut kept = 0;
        let mut mass = 0.0;
        for &id in &order {
            if mass >= self.top_p {
                break;
            }
            mass += probs[id as usize];
            kept += 1;
        }
        let order = &order[..kept.max(1)];

        let mut r = self.rng.next_f32() * mass;
        for &id in order {
            r -= probs[id as usize];
            if r < 0.0 {
                return id;
            }
        }
        *order.last().unwrap() // rounding left a sliver
    }
}

/// Index of the largest value; the first one on ties (like PyTorch's argmax).
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_first_maximum() {
        assert_eq!(Sampler::greedy().sample(&[1.0, 3.0, 3.0, -1.0]), 1);
    }

    #[test]
    fn small_top_p_keeps_only_the_top_token() {
        let mut s = Sampler::new(1.0, 0.01, 7);
        for _ in 0..100 {
            assert_eq!(s.sample(&[0.0, 2.0, 1.0]), 1);
        }
    }

    #[test]
    fn frequencies_follow_the_probabilities() {
        // ln(1), ln(3): probabilities 0.25 and 0.75.
        let logits = [0.0, 3f32.ln()];
        let mut s = Sampler::new(1.0, 1.0, 42);
        let ones = (0..20_000).filter(|_| s.sample(&logits) == 1).count();
        let frac = ones as f32 / 20_000.0;
        assert!((frac - 0.75).abs() < 0.02, "{frac}");
    }

    #[test]
    fn same_seed_same_choices() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let run = |seed| {
            let mut s = Sampler::new(0.9, 0.95, seed);
            (0..20).map(|_| s.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }
}
