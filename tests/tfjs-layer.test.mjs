import assert from "node:assert/strict";
import { decoderStep } from "../src/gemma3-decoder.js";
import { SessionCache } from "../src/gemma3-core.js";
import { TfjsLayerSession } from "../src/backends/tfjs-webgl/tfjs-layer.js";

// Lightweight tensor adapter checks Stage 11 control flow, dimensions,
// caching, and tensor disposal. It is NOT a real WebGL test.
let live = 0;
function tensor(values, shape) {
    live += 1;
    return {
        values: Float32Array.from(values),
        shape,
        matMul(other) {
            const [rows, columns] = this.shape;
            assert.equal(other.shape[0], columns);
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
        dispose() {
            live -= 1;
        }
    };
}
const tf = {
    tensor2d: tensor,
    tidy(callback) {
        const before = live;
        const result = callback();
        // Mimic automatic disposal of the input tensor in the tiny mock.
        assert.equal(live, before + 2);
        live -= 1;
        return result;
    }
};
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
    return Float32Array.from({ length: rows * columns }, (_, i) => Math.sin(i + seed) * 0.04);
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
const cpuCache = new SessionCache(1);
const gpu = new TfjsLayerSession(tf, layer, config);
for (let position = 0; position < 4; position += 1) {
    const input = Float32Array.from({ length: 8 }, (_, i) => Math.cos(i + position));
    const expected = decoderStep(input, layer, cpuCache, 0, position, config);
    const actual = await gpu.step(input);
    for (let i = 0; i < expected.length; i += 1) {
        assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6);
    }
    assert.equal(live, 7, "only cached weights should remain live");
}
gpu.dispose();
assert.equal(live, 0);
console.log("Stage 11 hybrid layer control-flow mock: PASS (not a WebGL GPU validation)");
