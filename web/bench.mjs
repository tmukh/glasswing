// Benchmark the browser build in Node (same V8 engine as Chrome), and check its output against
// the native CLI: greedy decoding must give the same story.
//
//   sh scripts/build-web.sh
//   node web/bench.mjs checkpoints/main/model-f16.safetensors [expected-text-file]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import init, { Engine } from "./pkg/glasswing_wasm.js";

const here = dirname(fileURLToPath(import.meta.url));
const [weightsPath, expectedPath] = process.argv.slice(2);

await init({ module_or_path: readFileSync(join(here, "pkg/glasswing_wasm_bg.wasm")) });
let t = performance.now();
const engine = new Engine(
  readFileSync(weightsPath),
  readFileSync(join(here, "../assets/tokenizer.json"), "utf8"),
);
console.log(`load: ${(performance.now() - t).toFixed(0)} ms`);

engine.reset(0n, 0, 1); // greedy
const prompt = "Once upon a time";
t = performance.now();
for (const id of engine.encode_prompt(prompt)) engine.step(id).free();
const prefill = performance.now() - t;

let text = prompt;
let n = 0;
t = performance.now();
while (n < 190 && engine.pos() < engine.max_seq_len()) {
  const id = engine.sample();
  if (id === engine.end_of_text()) break;
  text += engine.token_text(id);
  engine.step(id).free(); // traced step, as the visualization uses it
  n++;
}
const ms = performance.now() - t;
console.log(`prefill ${prefill.toFixed(0)} ms; ${n} tokens in ${ms.toFixed(0)} ms: ` +
  `${((n / ms) * 1000).toFixed(1)} tokens/s (traced steps)`);

if (expectedPath) {
  const expected = readFileSync(expectedPath, "utf8").replace(/\r?\n$/, "");
  const same = expected.startsWith(text) || text.startsWith(expected);
  console.log(same ? "same story as the native CLI" : `DIFFERENT:\n${text}\n---\n${expected}`);
  if (!same) process.exit(1);
}
