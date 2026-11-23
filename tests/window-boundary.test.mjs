import assert from "node:assert/strict";
import { compareWindowCaches } from "../src/gemma3-window-parity.js";
import { SessionCache, attentionForHead } from "../src/gemma3-core.js";

function weights(rows, columns, factor = 0.08) {
    return Float32Array.from({ length: rows * columns }, (_, i) => Math.sin(i + 1) * factor);
}

const config = {
    hiddenSize: 4,
    intermediateSize: 8,
    numLayers: 2,
    numHeads: 2,
    numKeyValueHeads: 1,
    headDim: 2,
    slidingWindow: 512,
    rmsNormEps: 1e-6,
    queryPreAttnScalar: 2,
    localRopeTheta: 10000,
    globalRopeTheta: 1000000,
    layerTypes: ["local", "global"],
    eosTokenIds: [99]
};

const layer = {
    inputLayernorm: new Float32Array(4),
    postAttentionLayernorm: new Float32Array(4),
    preFeedforwardLayernorm: new Float32Array(4),
    postFeedforwardLayernorm: new Float32Array(4),
    qNorm: new Float32Array(2),
    kNorm: new Float32Array(2),
    qProj: weights(4, 4),
    kProj: weights(2, 4),
    vProj: weights(2, 4),
    oProj: weights(4, 4),
    gateProj: weights(8, 4),
    upProj: weights(8, 4),
    downProj: weights(4, 8)
};

const inputs = Array.from({ length: 520 }, (_, position) =>
    Float32Array.from({ length: 4 }, (_, i) => Math.sin(position * 0.13 + i) * 0.4)
);

const result = compareWindowCaches(layer, inputs, config);
assert.equal(result.matches, true);
assert.equal(result.maximumError, 0);
assert.deepEqual(result.checkpoints.map(item => item.position), [0, 511, 512, 513, 519]);
assert.deepEqual(result.checkpoints.map(item => item.cacheFirstPosition), [0, 0, 1, 2, 8]);
assert.ok(result.checkpoints.every(item => item.cacheLength <= 512));

// Independently verify attention excludes the evicted position 0 at query 512.
// An out-of-window key has an enormous value; if accidentally included it
// would dominate the result. Global attention must still see that key.
const query = Float32Array.from([1, 0]);
const keys = [Float32Array.from([100, 0]), Float32Array.from([0, 1])];
const values = [Float32Array.from([1000, 0]), Float32Array.from([0, 2])];
const positions = [0, 512];
const local = attentionForHead(query, keys, values, positions, 512, "local", 512, 2);
const global = attentionForHead(query, keys, values, positions, 512, "global", 512, 2);
assert.deepEqual(Array.from(local), [0, 2]);
assert.ok(global[0] > 999);

// Confirm global cache retains all positions while local cache evicts.
const cache = new SessionCache(2);
for (let position = 0; position < 520; position += 1) {
    cache.append(0, position, [], [], "local", 512);
    cache.append(1, position, [], [], "global", 512);
}
assert.equal(cache.layers[0].positions.length, 512);
assert.equal(cache.layers[0].positions[0], 8);
assert.equal(cache.layers[1].positions.length, 520);
assert.equal(cache.layers[1].positions[0], 0);

console.log("Gemma 3 window boundary: synthetic 512-token test passed");
console.log(JSON.stringify(result, null, 4));
