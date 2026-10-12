//! Per-client rate limiting: a token bucket per IP address.
//!
//! Each client has a bucket holding up to `burst` tokens, refilled at `per_minute` tokens a minute.
//! A request takes one token; an empty bucket means 429 with the wait in `Retry-After`. Bursts are
//! allowed (a few quick messages), sustained load is not. Buckets live in memory: one replica,
//! and a restart forgetting them is harmless.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

pub struct RateLimiter {
    burst: f64,
    per_second: f64,
    buckets: Mutex<HashMap<String, (f64, Instant)>>,
}

impl RateLimiter {
    pub fn new(burst: u32, per_minute: u32) -> Self {
        RateLimiter {
            burst: burst as f64,
            per_second: per_minute as f64 / 60.0,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` if `client` may make a request now, else `Err(seconds until it may)`.
    pub fn check(&self, client: &str) -> Result<(), u64> {
        self.check_at(client, Instant::now())
    }

    fn check_at(&self, client: &str, now: Instant) -> Result<(), u64> {
        let mut buckets = self.buckets.lock().unwrap();
        if buckets.len() > 10_000 {
            // Forget clients whose buckets have refilled anyway, so the map can't grow forever.
            let (burst, rate) = (self.burst, self.per_second);
            buckets.retain(|_, (t, at)| *t + now.duration_since(*at).as_secs_f64() * rate < burst);
        }
        let (tokens, at) = buckets
            .entry(client.to_string())
            .or_insert((self.burst, now));
        *tokens =
            (*tokens + now.duration_since(*at).as_secs_f64() * self.per_second).min(self.burst);
        *at = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            Ok(())
        } else {
            Err(((1.0 - *tokens) / self.per_second).ceil() as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn burst_then_refill() {
        let rl = RateLimiter::new(3, 6); // one token every 10 s
        let t0 = Instant::now();
        for _ in 0..3 {
            assert!(rl.check_at("a", t0).is_ok());
        }
        assert_eq!(rl.check_at("a", t0), Err(10));
        assert!(rl.check_at("b", t0).is_ok(), "clients are independent");
        assert!(rl.check_at("a", t0 + Duration::from_secs(5)).is_err());
        assert!(rl.check_at("a", t0 + Duration::from_secs(11)).is_ok());
    }
}
