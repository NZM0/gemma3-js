import assert from "node:assert/strict";
import { Gemma3Generator } from "../src/gemma3-generate.js";

const config = {
    hiddenSize: 2,
    numLayers: 0,
    rmsNormEps: 1e-6,
    eosTokenIds: [1],
};
const weights = new Float32Array([0, 0, 1, 0, 0, 1]);
const generator = new Gemma3Generator([], new Float32Array(2), weights, config);
assert.ok(Math.abs(generator.embedding(1)[0] - Math.sqrt(2)) < 1e-6);
assert.equal(generator.embedding(1)[1], 0);
assert.deepEqual(generator.generate([1], 3), [1]);
assert.throws(() => generator.embedding(3), /Invalid token ID/);
console.log("Gemma 3 greedy generation synthetic test: PASS");
