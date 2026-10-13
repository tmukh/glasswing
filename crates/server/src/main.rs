//! glasswing-server: a public text-generation API on the glasswing engine.
//!
//!   GET  /                 a small chat page
//!   POST /v1/generate      {"model": "smollm2"|"glasswing", "messages"|"prompt", "max_tokens",
//!                           "temperature", "top_p", "seed"} -> Server-Sent Events
//!   GET  /healthz          liveness
//!   GET  /metrics          Prometheus (blocked at the Cloudflare Tunnel, scraped in-cluster)
//!
//! It is open to the internet without login, so it protects itself: a token bucket per client IP,
//! at most N generations at once (more wait briefly, then get 503), a cap on new tokens and on
//! wall-clock time, and generation stops as soon as the client disconnects. Prompts are never
//! logged.
//!
//! Configuration (environment): PORT, GLASSWING_SMOLLM2_DIR, GLASSWING_WEIGHTS,
//! GLASSWING_TOKENIZER, MAX_CONCURRENT (2), RATE_PER_MINUTE (6), RATE_BURST (5),
//! MAX_NEW_TOKENS (256), CONTEXT (1024), TIMEOUT_SECONDS (60).

mod generate;
mod limits;
mod metrics;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use generate::{Loaded, Message, Params, Piece};
use limits::RateLimiter;
use metrics::Metrics;

struct Config {
    max_new: usize,
    timeout: Duration,
}

struct App {
    smollm2: Option<Loaded>,
    glasswing: Option<Loaded>,
    gate: Arc<Semaphore>,
    limiter: RateLimiter,
    metrics: Metrics,
    cfg: Config,
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn log(fields: serde_json::Value) {
    // One JSON line per event, for Loki. Never prompts or full IPs.
    println!("{fields}");
}

#[tokio::main]
async fn main() {
    let context = env("CONTEXT", 1024usize);
    let smollm2 = std::env::var("GLASSWING_SMOLLM2_DIR").ok().map(|dir| {
        let t = Instant::now();
        let m = Loaded::smollm2(&dir, context).unwrap_or_else(|e| panic!("SmolLM2: {e}"));
        log(serde_json::json!({"event": "loaded", "model": "smollm2", "params": m.model.num_params(), "ms": t.elapsed().as_millis() as u64}));
        m
    });
    let glasswing = match (
        std::env::var("GLASSWING_WEIGHTS"),
        std::env::var("GLASSWING_TOKENIZER"),
    ) {
        (Ok(w), Ok(t)) => {
            Some(Loaded::glasswing(&w, &t).unwrap_or_else(|e| panic!("glasswing: {e}")))
        }
        _ => None,
    };
    assert!(
        smollm2.is_some() || glasswing.is_some(),
        "no model configured"
    );

    let app = Arc::new(App {
        smollm2,
        glasswing,
        gate: Arc::new(Semaphore::new(env("MAX_CONCURRENT", 2usize))),
        limiter: RateLimiter::new(env("RATE_BURST", 5u32), env("RATE_PER_MINUTE", 6u32)),
        metrics: Metrics::new(),
        cfg: Config {
            max_new: env("MAX_NEW_TOKENS", 256usize),
            timeout: Duration::from_secs(env("TIMEOUT_SECONDS", 60u64)),
        },
    });

    // The chat page is same-origin; takodana.dev may call the API from the browser too.
    let cors = CorsLayer::new()
        .allow_origin(HeaderValue::from_static("https://takodana.dev"))
        .allow_methods([Method::POST])
        .allow_headers([header::CONTENT_TYPE]);
    let router = Router::new()
        .route("/", get(|| async { Html(include_str!("chat.html")) }))
        .route("/healthz", get(|| async { "ok\n" }))
        .route("/metrics", get(metrics_handler))
        .route("/v1/generate", post(generate_handler).layer(cors))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .with_state(app);

    let port = env("PORT", 8080u16);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .unwrap();
    log(serde_json::json!({"event": "listening", "port": port}));
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
    .unwrap();
}

async fn metrics_handler(State(app): State<Arc<App>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        app.metrics.render(),
    )
}

#[derive(Deserialize)]
struct Request {
    model: String,
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
}

fn error(status: StatusCode, msg: &str) -> Response {
    (status, Json(serde_json::json!({ "error": msg }))).into_response()
}

/// The client's address. Behind the Cloudflare Tunnel every request carries CF-Connecting-IP;
/// the only way in is the tunnel (NetworkPolicy), so the header can be trusted.
fn client_ip(headers: &HeaderMap, peer: SocketAddr) -> String {
    headers
        .get("cf-connecting-ip")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| peer.ip().to_string())
}

async fn generate_handler(
    State(app): State<Arc<App>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> Response {
    let started = Instant::now();
    let m = &app.metrics;
    let model_name = req.model.as_str();
    let loaded = match model_name {
        "smollm2" => app.smollm2.as_ref(),
        "glasswing" => app.glasswing.as_ref(),
        _ => None,
    };
    let Some(loaded) = loaded else {
        return error(
            StatusCode::BAD_REQUEST,
            "model must be \"smollm2\" or \"glasswing\"",
        );
    };
    let name = loaded.name;

    // Validate before spending anything.
    let total_chars: usize =
        req.messages.iter().map(|x| x.content.len()).sum::<usize>() + req.prompt.len();
    let bad = if req.messages.len() > 20 {
        Some("at most 20 messages")
    } else if total_chars > 8000 {
        Some("at most 8000 characters of input")
    } else if req
        .messages
        .iter()
        .any(|x| x.role != "user" && x.role != "assistant")
    {
        Some("message roles must be user or assistant")
    } else if req.messages.is_empty() && req.prompt.trim().is_empty() {
        Some("give messages or a prompt")
    } else {
        None
    };
    if let Some(msg) = bad {
        m.requests.with_label_values(&[name, "bad_request"]).inc();
        return error(StatusCode::BAD_REQUEST, msg);
    }

    if let Err(wait) = app.limiter.check(&client_ip(&headers, peer)) {
        m.requests.with_label_values(&[name, "rate_limited"]).inc();
        m.rate_limited.inc();
        let mut r = error(
            StatusCode::TOO_MANY_REQUESTS,
            "too many requests; try again shortly",
        );
        r.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(wait));
        return r;
    }

    let params = Params {
        max_new: req
            .max_tokens
            .unwrap_or(app.cfg.max_new)
            .clamp(1, app.cfg.max_new),
        temperature: req.temperature.unwrap_or(0.3).clamp(0.0, 2.0),
        top_p: req.top_p.unwrap_or(0.9).clamp(0.05, 1.0),
        seed: req
            .seed
            .unwrap_or_else(|| started.elapsed().as_nanos() as u64 ^ 0x5eed),
        timeout: app.cfg.timeout,
    };
    let ids = match generate::prompt_ids(loaded, &req.messages, &req.prompt, params.max_new) {
        Ok(ids) => ids,
        Err(msg) => {
            m.requests.with_label_values(&[name, "bad_request"]).inc();
            return error(StatusCode::BAD_REQUEST, &msg);
        }
    };

    // At most MAX_CONCURRENT generations; others wait up to 5 s for a slot.
    let permit = match tokio::time::timeout(
        Duration::from_secs(5),
        app.gate.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        _ => {
            m.requests.with_label_values(&[name, "busy"]).inc();
            let mut r = error(
                StatusCode::SERVICE_UNAVAILABLE,
                "busy; try again in a few seconds",
            );
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(5));
            return r;
        }
    };

    let (tx, rx) = mpsc::channel::<Piece>(64);
    let app2 = app.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit; // released when generation ends
        let loaded = match name {
            "smollm2" => app2.smollm2.as_ref().unwrap(),
            _ => app2.glasswing.as_ref().unwrap(),
        };
        let m = &app2.metrics;
        m.active.inc();
        let prompt_tokens = ids.len();
        let (reason, n, ttft) = generate::run(loaded, ids, &params, started, &tx);
        m.active.dec();
        let total = started.elapsed();
        let decode = ttft.map(|t| total.saturating_sub(t)).unwrap_or(total);
        let tps = if n > 1 {
            (n - 1) as f64 / decode.as_secs_f64().max(1e-9)
        } else {
            0.0
        };
        m.tokens.with_label_values(&[name]).inc_by(n as u64);
        if let Some(t) = ttft {
            m.ttft.with_label_values(&[name]).observe(t.as_secs_f64());
        }
        if n > 1 {
            m.tokens_per_second.with_label_values(&[name]).observe(tps);
        }
        let outcome = match reason {
            "stop" | "length" => "ok",
            other => other,
        };
        m.requests.with_label_values(&[name, outcome]).inc();
        let ttft_ms = ttft.map_or(0, |t| t.as_millis() as u64);
        log(
            serde_json::json!({"event": "generation", "model": name, "outcome": outcome, "reason": reason,
            "prompt_tokens": prompt_tokens, "tokens": n, "ttft_ms": ttft_ms, "tokens_per_s": (tps * 10.0).round() / 10.0,
            "ms": total.as_millis() as u64}),
        );
        let _ = tx.blocking_send(Piece::Done {
            reason,
            tokens: n,
            ttft_ms,
            tokens_per_s: tps,
        });
    });

    let stream = ReceiverStream::new(rx).map(|piece| {
        let ev = match piece {
            Piece::Text(t) => Event::default().json_data(serde_json::json!({ "text": t })),
            Piece::Done {
                reason,
                tokens,
                ttft_ms,
                tokens_per_s,
            } => Event::default().event("done").json_data(
                serde_json::json!({ "reason": reason, "tokens": tokens, "ttft_ms": ttft_ms,
                    "tokens_per_s": (tokens_per_s * 10.0).round() / 10.0 }),
            ),
            Piece::Error(e) => Event::default()
                .event("error")
                .json_data(serde_json::json!({ "error": e })),
        };
        Ok::<_, Infallible>(ev.unwrap())
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
