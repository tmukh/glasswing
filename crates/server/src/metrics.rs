//! Prometheus metrics. Scraped in-cluster at /metrics; the Cloudflare Tunnel answers 404 for that
//! path, so it isn't public.

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
    TextEncoder,
};

pub struct Metrics {
    registry: Registry,
    /// Requests by model and outcome: ok, client_gone, timeout, rate_limited, busy, bad_request.
    pub requests: IntCounterVec,
    pub tokens: IntCounterVec,
    /// Time from accepting a request to its first generated token (prompt processing included).
    pub ttft: HistogramVec,
    /// Decode speed of each finished generation.
    pub tokens_per_second: HistogramVec,
    pub active: IntGauge,
    pub rate_limited: IntCounter,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let requests = IntCounterVec::new(
            Opts::new(
                "glasswing_requests_total",
                "Generation requests by model and outcome",
            ),
            &["model", "outcome"],
        )
        .unwrap();
        let tokens = IntCounterVec::new(
            Opts::new("glasswing_tokens_generated_total", "Tokens generated"),
            &["model"],
        )
        .unwrap();
        let ttft = HistogramVec::new(
            HistogramOpts::new(
                "glasswing_time_to_first_token_seconds",
                "Time to first token",
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]),
            &["model"],
        )
        .unwrap();
        let tokens_per_second = HistogramVec::new(
            HistogramOpts::new("glasswing_tokens_per_second", "Decode speed per generation")
                .buckets(vec![2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]),
            &["model"],
        )
        .unwrap();
        let active =
            IntGauge::new("glasswing_active_generations", "Generations running now").unwrap();
        let rate_limited = IntCounter::new(
            "glasswing_rate_limited_total",
            "Requests refused by the per-IP limit",
        )
        .unwrap();
        registry.register(Box::new(requests.clone())).unwrap();
        registry.register(Box::new(tokens.clone())).unwrap();
        registry.register(Box::new(ttft.clone())).unwrap();
        registry
            .register(Box::new(tokens_per_second.clone()))
            .unwrap();
        registry.register(Box::new(active.clone())).unwrap();
        registry.register(Box::new(rate_limited.clone())).unwrap();
        Metrics {
            registry,
            requests,
            tokens,
            ttft,
            tokens_per_second,
            active,
            rate_limited,
        }
    }

    pub fn render(&self) -> String {
        let mut out = Vec::new();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut out)
            .unwrap();
        String::from_utf8(out).unwrap()
    }
}
