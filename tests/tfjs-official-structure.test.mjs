import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { GEMMA3_1B } from "../src/gemma3-core.js";
import { TfjsOfficialModel } from "../src/backends/tfjs-webgl/tfjs-official-model.js";
import { TfjsOfficialHead } from "../src/backends/tfjs-webgl/tfjs-official-head.js";

assert.equal(GEMMA3_1B.numLayers, 26);
assert.equal(GEMMA3_1B.slidingWindow, 512);
assert.equal(GEMMA3_1B.layerTypes.filter(type => type === "global").length, 4);
assert.equal(typeof TfjsOfficialModel.prototype.prefill, "function");
assert.equal(typeof TfjsOfficialModel.prototype.step, "function");
assert.equal(typeof TfjsOfficialHead.prototype.predict, "function");
const source = readFileSync(new URL("../src/backends/tfjs-webgl/tfjs-official-model.js", import.meta.url), "utf8");
assert.match(source, /\.matMul\(/);
assert.match(source, /tf\.softmax\(/);
assert.match(source, /queryPosition - c\.slidingWindow/);
console.log("TF.js independent model structure: PASS (not numerical WebGL parity)");
