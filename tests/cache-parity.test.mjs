import assert from "node:assert/strict";
import { Gemma3Generator } from "../src/gemma3-generate.js";
import { compareCachedAndRecomputed } from "../src/gemma3-cache-parity.js";

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
    layerTypes: ["local", "global"],
    eosTokenIds: [99]
};

function weights(rows, columns, factor = 0.08) {
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

const embeddingTable = weights(6, 4, 0.2);
const generator = new Gemma3Generator([layer(), layer()], new Float32Array(4), embeddingTable, config);
const result = compareCachedAndRecomputed(generator, [1, 2, 3, 4], 4, 1e-6);

assert.equal(result.matches, true);
assert.equal(result.steps.length, 4);
assert.deepEqual(result.steps.map(item => item.prefixLength), [4, 5, 6, 7]);
assert.ok(result.steps.every(item => item.hiddenError.maxAbsoluteError === 0));
assert.ok(result.steps.every(item => item.logitError.maxAbsoluteError === 0));
assert.deepEqual(generator.generate([1, 2, 3, 4], 4), result.generatedTokenIds);
assert.throws(() => compareCachedAndRecomputed(generator, [], 2), /Prompt/);

console.log("Gemma 3 KV cache parity: synthetic tests passed (local window crossed)");
