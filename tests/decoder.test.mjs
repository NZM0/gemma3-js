import assert from "node:assert/strict";
import { decoderStep, DecoderSession, linear } from "../src/gemma3-decoder.js";
import { SessionCache } from "../src/gemma3-core.js";

const config = {
    hiddenSize: 4,
    intermediateSize: 8,
    numLayers: 2,
    numHeads: 2,
    numKeyValueHeads: 1,
    headDim: 2,
    slidingWindow: 2,
    rmsNormEps: 1e-6,
    queryPreAttnScalar: 2,
    localRopeTheta: 10000,
    globalRopeTheta: 1000000,
    layerTypes: ["local", "global"]
};

function weights(rows, columns, factor = 0.01) {
    return Float32Array.from({ length: rows * columns }, (_, i) => Math.sin(i + 1) * factor);
}

function layer() {
    return {
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
}

assert.deepEqual(Array.from(linear([2, 3], Float32Array.from([1, 0, 0, 2]), 2, 2)), [2, 6]);
assert.throws(() => linear([1], new Float32Array(4), 2, 2), /dimension mismatch/);

const layers = [layer(), layer()];
const session = new DecoderSession(config, layers);
const first = session.step([1, 2, 3, 4]);
assert.equal(first.length, 4);
assert.ok(first.every(Number.isFinite));

for (let i = 0; i < 4; i += 1) {
    session.step([1, 2, 3, 4]);
}

assert.deepEqual(session.cache.layers[0].positions, [3, 4]);
assert.deepEqual(session.cache.layers[1].positions, [0, 1, 2, 3, 4]);
session.reset();
assert.equal(session.position, 0);
assert.equal(session.cache.layers[1].positions.length, 0);
assert.deepEqual(Array.from(session.step([1, 2, 3, 4])), Array.from(first));

const other = new SessionCache(2);
decoderStep([1, 2, 3, 4], layers[0], other, 0, 0, config);
assert.equal(other.layers[0].positions.length, 1);
assert.equal(session.cache.layers[0].positions.length, 1);
console.log("Gemma 3 decoder: synthetic tests passed");
