import { GEMMA3_1B } from "../../gemma3-core.js";

/** Independent TensorFlow.js transcription of the Google Gemma 3 1B text decoder.
 * All decoder arithmetic, including attention, runs as TensorFlow.js tensor ops.
 * The cache is maintained per layer; CPU reads are confined to the final output.
 */
export class TfjsOfficialModel {
    constructor(tf, layers, finalNorm, config = GEMMA3_1B) {
        if (layers.length !== config.numLayers) {
            throw new Error(`Expected ${config.numLayers} layers, got ${layers.length}`);
        }
        this.tf = tf;
        this.config = config;
        this.position = 0;
        this.disposed = false;
        this.layers = [];
        this.finalNorm = null;
        try {
            for (let index = 0; index < layers.length; index += 1) {
                this.layers.push(this.createLayer(layers[index], index));
            }
            this.finalNorm = tf.tensor1d(finalNorm);
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    createLayer(layer, index) {
        const tf = this.tf;
        const c = this.config;
        const shapes = {
            qProj: [c.numHeads * c.headDim, c.hiddenSize],
            kProj: [c.numKeyValueHeads * c.headDim, c.hiddenSize],
            vProj: [c.numKeyValueHeads * c.headDim, c.hiddenSize],
            oProj: [c.hiddenSize, c.numHeads * c.headDim],
            gateProj: [c.intermediateSize, c.hiddenSize],
            upProj: [c.intermediateSize, c.hiddenSize],
            downProj: [c.hiddenSize, c.intermediateSize]
        };
        const weights = {};
        const norms = {};
        try {
            for (const [name, shape] of Object.entries(shapes)) {
                if (layer[name].length !== shape[0] * shape[1]) {
                    throw new Error(`Layer ${index} ${name}: weight shape mismatch`);
                }
                weights[name] = tf.tensor2d(layer[name], shape);
            }
            for (const name of ["inputLayernorm", "qNorm", "kNorm", "postAttentionLayernorm", "preFeedforwardLayernorm", "postFeedforwardLayernorm"]) {
                norms[name] = tf.tensor1d(layer[name]);
            }
            return { index, weights, norms, keys: null, values: null };
        } catch (error) {
            Object.values(weights).forEach(tensor => tensor.dispose());
            Object.values(norms).forEach(tensor => tensor.dispose());
            throw error;
        }
    }

    norm(x, weight) {
        const tf = this.tf;
        const last = x.shape.length - 1;
        return x.mul(tf.rsqrt(x.square().mean(last, true).add(this.config.rmsNormEps))).mul(weight.add(1));
    }

    linear(x, weights) {
        return x.matMul(weights, false, true);
    }

    rope(x, positions, theta) {
        const tf = this.tf;
        const half = this.config.headDim / 2;
        const frequencies = Float32Array.from({ length: half }, (_, i) => 1 / theta ** (2 * i / this.config.headDim));
        const inverse = tf.tensor1d(frequencies);
        const angles = tf.tensor1d(positions).expandDims(1).mul(inverse.expandDims(0));
        const cos = angles.cos().expandDims(1);
        const sin = angles.sin().expandDims(1);
        const [left, right] = tf.split(x, 2, -1);
        return tf.concat([left.mul(cos).sub(right.mul(sin)), right.mul(cos).add(left.mul(sin))], -1);
    }

    gelu(x) {
        const tf = this.tf;
        return x.mul(0.5).mul(tf.tanh(x.add(x.pow(tf.scalar(3)).mul(0.044715)).mul(Math.sqrt(2 / Math.PI))).add(1));
    }

    /** Inputs are embeddings [T,H], already scaled by sqrt(hidden_size).
     * Batching uses causal and sliding-window masks, with layer-local cache state.
     */
    async forward(embeddings) {
        if (this.disposed) {
            throw new Error("TfjsOfficialModel is disposed");
        }
        const tf = this.tf;
        const c = this.config;
        const tokens = embeddings.length / c.hiddenSize;
        if (!Number.isInteger(tokens) || tokens < 1 || this.position + tokens > c.maxPositions) {
            throw new Error("Invalid input length or context overflow");
        }
        const start = this.position;
        let hidden = tf.tensor2d(embeddings, [tokens, c.hiddenSize]);
        try {
            for (const layer of this.layers) {
                const { next, keys, values } = tf.tidy(() => {
                    const w = layer.weights;
                    const n = layer.norms;
                    const normed = this.norm(hidden, n.inputLayernorm);
                    const qRaw = this.linear(normed, w.qProj).reshape([tokens, c.numHeads, c.headDim]);
                    const kRaw = this.linear(normed, w.kProj).reshape([tokens, c.numKeyValueHeads, c.headDim]);
                    const vNew = this.linear(normed, w.vProj).reshape([tokens, c.numKeyValueHeads, c.headDim]);
                    const theta = c.layerTypes[layer.index] === "local" ? c.localRopeTheta : c.globalRopeTheta;
                    const positions = Array.from({ length: tokens }, (_, i) => start + i);
                    const q = this.rope(this.norm(qRaw, n.qNorm), positions, theta);
                    const kNew = this.rope(this.norm(kRaw, n.kNorm), positions, theta);
                    const kAll = layer.keys ? tf.concat([layer.keys, kNew], 0) : kNew;
                    const vAll = layer.values ? tf.concat([layer.values, vNew], 0) : vNew;
                    const keyStart = layer.keys ? start - layer.keys.shape[0] : start;
                    const keyCount = kAll.shape[0];
                    const groups = c.numHeads / c.numKeyValueHeads;
                    const kHeads = tf.tile(kAll.transpose([1, 0, 2]), [groups, 1, 1]);
                    const vHeads = tf.tile(vAll.transpose([1, 0, 2]), [groups, 1, 1]);
                    const scores = q.transpose([1, 0, 2]).matMul(kHeads, false, true).div(Math.sqrt(c.queryPreAttnScalar));
                    const mask = new Float32Array(tokens * keyCount);
                    for (let i = 0; i < tokens; i += 1) {
                        const queryPosition = start + i;
                        for (let j = 0; j < keyCount; j += 1) {
                            const keyPosition = keyStart + j;
                            const allowed = keyPosition <= queryPosition && (c.layerTypes[layer.index] !== "local" || keyPosition > queryPosition - c.slidingWindow);
                            mask[i * keyCount + j] = allowed ? 0 : -1e9;
                        }
                    }
                    const attention = tf.softmax(scores.add(tf.tensor2d(mask, [tokens, keyCount]).expandDims(0)), -1);
                    const context = attention.matMul(vHeads).transpose([1, 0, 2]).reshape([tokens, c.numHeads * c.headDim]);
                    const projected = this.linear(context, w.oProj);
                    const afterAttention = hidden.add(this.norm(projected, n.postAttentionLayernorm));
                    const ffInput = this.norm(afterAttention, n.preFeedforwardLayernorm);
                    const gated = this.gelu(this.linear(ffInput, w.gateProj)).mul(this.linear(ffInput, w.upProj));
                    const next = afterAttention.add(this.norm(this.linear(gated, w.downProj), n.postFeedforwardLayernorm));
                    // Local cache keeps exactly the most recent 512 keys, including the current key.
                    const keep = c.layerTypes[layer.index] === "local" ? Math.min(c.slidingWindow, keyCount) : keyCount;
                    const keys = kAll.slice([keyCount - keep, 0, 0], [keep, -1, -1]);
                    const values = vAll.slice([keyCount - keep, 0, 0], [keep, -1, -1]);
                    return { next, keys, values };
                });
                hidden.dispose();
                layer.keys?.dispose();
                layer.values?.dispose();
                layer.keys = keys;
                layer.values = values;
                hidden = next;
            }
            this.position += tokens;
            const result = tf.tidy(() => {
                const last = hidden.slice([tokens - 1, 0], [1, -1]);
                return this.norm(last, this.finalNorm).reshape([c.hiddenSize]);
            });
            return { lastNormalized: result, output: result };
        } finally {
            hidden.dispose();
        }
    }

    // Independent no-cache path: recompute the entire prefix from scratch.
    // The returned final-normalized tensor remains owned by the caller.
    async forwardNoCache(embeddings) {
        this.reset();
        try {
            const flat = Float32Array.from(embeddings.flatMap(value => Array.from(value)));
            return await this.forward(flat);
        } finally {
            this.reset();
        }
    }

    async step(embedding) {
        return this.forward(embedding);
    }

    async prefill(embeddings) {
        if (embeddings.length === 0) {
            throw new Error("Empty prefill");
        }
        const flat = Float32Array.from(embeddings.flatMap(value => Array.from(value)));
        return this.forward(flat);
    }

    reset() {
        this.position = 0;
        for (const layer of this.layers) {
            layer.keys?.dispose();
            layer.values?.dispose();
            layer.keys = null;
            layer.values = null;
        }
    }

    dispose() {
        if (this.disposed) {
            return;
        }
        this.reset();
        for (const layer of this.layers) {
            Object.values(layer.weights).forEach(tensor => tensor.dispose());
            Object.values(layer.norms).forEach(tensor => tensor.dispose());
        }
        this.finalNorm?.dispose();
        this.disposed = true;
    }
}
