import { rmsNorm, geluTanh, applyRopeHalfSplit, attentionForHead, SessionCache, GEMMA3_1B } from "../../gemma3-core.js";

const projectionNames = ["qProj", "kProj", "vProj", "oProj", "gateProj", "upProj", "downProj"];

function splitHeads(vector, count, size) {
    return Array.from({ length: count }, (_, index) => vector.slice(index * size, (index + 1) * size));
}

function add(a, b) {
    return Float32Array.from(a, (value, index) => value + b[index]);
}

/** Stage 11 hybrid: WebGL projection matvec; CPU RMSNorm/RoPE/attention/GELU.
 * The seven weight tensors are uploaded once and reused across tokens.
 */
export class TfjsLayerSession {
    constructor(tf, layer, config = GEMMA3_1B, layerIndex = 0) {
        this.tf = tf;
        this.layer = layer;
        this.config = config;
        this.layerIndex = layerIndex;
        this.position = 0;
        this.cache = new SessionCache(config.numLayers);
        this.weights = {};
        this.disposed = false;

        const shapes = {
            qProj: [config.numHeads * config.headDim, config.hiddenSize],
            kProj: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
            vProj: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
            oProj: [config.hiddenSize, config.numHeads * config.headDim],
            gateProj: [config.intermediateSize, config.hiddenSize],
            upProj: [config.intermediateSize, config.hiddenSize],
            downProj: [config.hiddenSize, config.intermediateSize]
        };

        try {
            for (const name of projectionNames) {
                const [rows, columns] = shapes[name];
                if (layer[name].length !== rows * columns) {
                    throw new Error(`${name}: incompatible weight shape`);
                }
                this.weights[name] = tf.tensor2d(layer[name], [rows, columns], "float32");
            }
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    async project(name, input) {
        const tf = this.tf;
        const result = tf.tidy(() => {
            const vector = tf.tensor2d(input, [input.length, 1], "float32");
            return this.weights[name].matMul(vector).reshape([this.weights[name].shape[0]]);
        });

        try {
            return Float32Array.from(await result.data());
        } finally {
            result.dispose();
        }
    }

    async step(input) {
        if (this.disposed) {
            throw new Error("TfjsLayerSession has been disposed");
        }
        const config = this.config;
        const layer = this.layer;
        const position = this.position;
        const index = this.layerIndex;
        const layerType = config.layerTypes[index];
        const theta = layerType === "local" ? config.localRopeTheta : config.globalRopeTheta;
        const normalized = rmsNorm(input, layer.inputLayernorm, config.rmsNormEps);

        const [qFlat, kFlat, vFlat] = await Promise.all([
            this.project("qProj", normalized),
            this.project("kProj", normalized),
            this.project("vProj", normalized)
        ]);
        const q = splitHeads(qFlat, config.numHeads, config.headDim);
        const k = splitHeads(kFlat, config.numKeyValueHeads, config.headDim);
        const v = splitHeads(vFlat, config.numKeyValueHeads, config.headDim);
        const queries = q.map(head => applyRopeHalfSplit(rmsNorm(head, layer.qNorm, config.rmsNormEps), position, theta));
        const keys = k.map(head => applyRopeHalfSplit(rmsNorm(head, layer.kNorm, config.rmsNormEps), position, theta));

        this.cache.append(index, position, keys, v, layerType, config.slidingWindow);
        const state = this.cache.layers[index];
        const heads = queries.map((query, qIndex) => {
            const kvIndex = Math.floor(qIndex / (config.numHeads / config.numKeyValueHeads));
            return attentionForHead(
                query,
                state.keys.map(entry => entry[kvIndex]),
                state.values.map(entry => entry[kvIndex]),
                state.positions,
                position,
                layerType,
                config.slidingWindow,
                config.queryPreAttnScalar
            );
        });
        const combined = Float32Array.from(heads.flatMap(head => Array.from(head)));
        const attention = await this.project("oProj", combined);
        const afterAttention = add(input, rmsNorm(attention, layer.postAttentionLayernorm, config.rmsNormEps));
        const ffInput = rmsNorm(afterAttention, layer.preFeedforwardLayernorm, config.rmsNormEps);
        const [gate, up] = await Promise.all([
            this.project("gateProj", ffInput),
            this.project("upProj", ffInput)
        ]);
        const activated = Float32Array.from(gate, (value, i) => geluTanh(value) * up[i]);
        const down = await this.project("downProj", activated);
        const output = add(afterAttention, rmsNorm(down, layer.postFeedforwardLayernorm, config.rmsNormEps));
        this.position += 1;
        return output;
    }

    reset() {
        this.cache.clear();
        this.position = 0;
    }

    dispose() {
        for (const tensor of Object.values(this.weights)) {
            tensor.dispose();
        }
        this.weights = {};
        this.disposed = true;
    }
}
