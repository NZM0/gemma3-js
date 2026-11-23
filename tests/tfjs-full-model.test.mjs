import assert from "node:assert/strict";
import { TfjsFullModelSession } from "../src/backends/tfjs-webgl/tfjs-full-model.js";
import { Gemma3Model } from "../src/gemma3-model.js";

const config = {
    hiddenSize: 8,
    intermediateSize: 12,
    numLayers: 1,
    numHeads: 2,
    numKeyValueHeads: 1,
    headDim: 4,
    layerTypes: ["local"],
    localRopeTheta: 10000,
    globalRopeTheta: 1000000,
    rmsNormEps: 1e-6,
    queryPreAttnScalar: 4,
    slidingWindow: 2
};

function matrix(rows, columns, seed) {
    return Float32Array.from({ length: rows * columns }, (_, index) => Math.sin(index + seed) * 0.04);
}

const layer = {
    qProj: matrix(8, 8, 1),
    kProj: matrix(4, 8, 2),
    vProj: matrix(4, 8, 3),
    oProj: matrix(8, 8, 4),
    gateProj: matrix(12, 8, 5),
    upProj: matrix(12, 8, 6),
    downProj: matrix(8, 12, 7),
    inputLayernorm: new Float32Array(8),
    qNorm: new Float32Array(4),
    kNorm: new Float32Array(4),
    postAttentionLayernorm: new Float32Array(8),
    preFeedforwardLayernorm: new Float32Array(8),
    postFeedforwardLayernorm: new Float32Array(8)
};

function tensor(values, shape) {
    return {
        values: Float32Array.from(values),
        shape,
        matMul(other) {
            const [rows, columns] = this.shape;
            const result = new Float32Array(rows);
            for (let row = 0; row < rows; row += 1) {
                for (let column = 0; column < columns; column += 1) {
                    result[row] += this.values[row * columns + column] * other.values[column];
                }
            }
            return tensor(result, [rows, 1]);
        },
        reshape(nextShape) {
            this.shape = nextShape;
            return this;
        },
        async data() {
            return this.values;
        },
        dispose() {}
    };
}

const tf = {
    tensor2d: tensor,
    tidy(callback) {
        return callback();
    }
};

const finalNorm = new Float32Array(config.hiddenSize);
const reference = new Gemma3Model([layer], finalNorm, config);
const hybrid = new TfjsFullModelSession(tf, [layer], finalNorm, config);
const vectors = Array.from({ length: 5 }, (_, position) =>
    Float32Array.from({ length: config.hiddenSize }, (_, index) => Math.cos(index + position))
);
const embeddings = Float32Array.from(vectors.flatMap(vector => Array.from(vector)));
const expected = reference.forwardEmbeddings(embeddings);

for (let position = 0; position < vectors.length; position += 1) {
    const result = await hybrid.step(vectors[position]);
    const referenceHidden = expected.hidden.subarray(position * config.hiddenSize, (position + 1) * config.hiddenSize);
    for (let index = 0; index < config.hiddenSize; index += 1) {
        assert.ok(Math.abs(result.output[index] - referenceHidden[index]) < 1e-6);
    }
}
assert.equal(hybrid.position, 5);
hybrid.reset();
assert.equal(hybrid.position, 0);
const replay = await hybrid.forwardNoCache(vectors);
for (let index = 0; index < config.hiddenSize; index += 1) {
    assert.ok(Math.abs(replay.lastNormalized[index] - expected.lastNormalized[index]) < 1e-6);
}
assert.equal(hybrid.position, 0);
hybrid.dispose();
console.log("TF.js full-model hybrid adapter: PASS (mock; not browser WebGL)");
