// glasswing Web Worker: runs the model off the main thread, so drawing never waits on math.
//
// Messages in:
//   {type: "load", base}                    fetch base/glasswing_wasm_bg.wasm, model.bin, tokenizer.json
//   {type: "generate", prompt, maxTokens, temperature, topP, seed}
//   {type: "stop"}
// Messages out:
//   {type: "progress", loaded, total}       weights download progress (bytes)
//   {type: "ready", config, version}
//   {type: "step", ...}                     one token: text, attention, mlp, resid, top-k
//   {type: "done", reason}                  "end" | "length" | "stopped"
//   {type: "error", message}
//
// Typed arrays are transferred, not copied.

import init, { Engine, version } from "./glasswing_wasm.js";

let engine = null;
let stopRequested = false;

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

async function load(base) {
  await init({ module_or_path: `${base}/glasswing_wasm_bg.wasm` });
  const [weights, tokenizer] = await Promise.all([
    fetchWithProgress(`${base}/model.bin`),
    fetch(`${base}/tokenizer.json`).then((r) => r.text()),
  ]);
  engine = new Engine(weights, tokenizer);
  postMessage({ type: "ready", config: JSON.parse(engine.config_json()), version: version() });
}

function send(step, isPrompt) {
  const msg = {
    type: "step",
    prompt: isPrompt,
    token: step.token(),
    text: engine.token_text(step.token()),
    pos: step.pos(),
    attention: step.attention(),
    mlp: step.mlp(),
    resid: step.resid(),
    topIds: step.top_ids(),
    topProbs: step.top_probs(),
  };
  msg.topTexts = Array.from(msg.topIds, (id) => engine.token_text(id));
  step.free();
  postMessage(msg, [msg.attention.buffer, msg.mlp.buffer, msg.resid.buffer]);
}

// Yield to the event loop between tokens so a "stop" message can arrive.
const tick = () => new Promise((r) => setTimeout(r, 0));

async function generate({ prompt, maxTokens, temperature, topP, seed }) {
  stopRequested = false;
  engine.reset(BigInt(seed), temperature, topP);
  const ids = engine.encode_prompt(prompt);
  for (const id of ids) send(engine.step(id), true);
  for (let n = 0; n < maxTokens; n++) {
    await tick();
    if (stopRequested) return postMessage({ type: "done", reason: "stopped" });
    if (engine.pos() >= engine.max_seq_len()) return postMessage({ type: "done", reason: "length" });
    const next = engine.sample();
    if (next === engine.end_of_text()) return postMessage({ type: "done", reason: "end" });
    send(engine.step(next), false);
  }
  postMessage({ type: "done", reason: "length" });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.base);
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopRequested = true;
  } catch (e) {
    postMessage({ type: "error", message: String(e?.message ?? e) });
  }
};
