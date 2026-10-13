#!/usr/bin/env sh
# Build the browser bundle into web/pkg: the wasm module plus wasm-bindgen's JS glue.
# Needs: rustup target wasm32-unknown-unknown (rust-toolchain.toml), wasm-bindgen-cli matching
# the wasm-bindgen crate version in Cargo.lock.
set -eu
cd "$(dirname "$0")/.."
cargo build -p glasswing-wasm --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir web/pkg --no-typescript \
  target/wasm32-unknown-unknown/release/glasswing_wasm.wasm
ls -l web/pkg
