import { rmsNorm, geluTanh, applyRopeHalfSplit, attentionForHead, SessionCache } from "./gemma3-core.js";

// Debug-oriented reference: one token at a time, float32, row-major [out, in].
// The checkpoint's Gemma RMSNorm parameters are deltas from 1.
export function linear(vector, matrix, outFeatures, inFeatures) {
    if (vector.length !== inFeatures || matrix.length !== outFeatures * inFeatures) {
        throw new Error("linear: dimension mismatch");
    }

    const result = new Float32Array(outFeatures);

    for (let out = 0; out < outFeatures; out += 1) {
        let sum = 0;

        for (let input = 0; input < inFeatures; input += 1) {
            sum += matrix[out * inFeatures + input] * vector[input];
        }

        result[out] = sum;
    }

    return result;
}

function add(a, b) {
    if (a.length !== b.length) {
        throw new Error("add: dimension mismatch");
    }

    return Float32Array.from(a, (value, i) => value + b[i]);
}

function splitHeads(vector, heads, headDim) {
    if (vector.length !== heads * headDim) {
        throw new Error("splitHeads: dimension mismatch");
    }

    return Array.from({ length: heads }, (_, i) => vector.slice(i * headDim, (i + 1) * headDim));
}

export function decoderStep(input, layer, cache, layerIndex, position, config) {
    const hidden = config.hiddenSize;
    const headDim = config.headDim;
    const qHeads = config.numHeads;
    const kvHeads = config.numKeyValueHeads;
    const intermediate = config.intermediateSize;
    const layerType = config.layerTypes[layerIndex];
    const theta = layerType === "local" ? config.localRopeTheta : config.globalRopeTheta;

    if (!layerType || qHeads % kvHeads !== 0) {
        throw new Error("decoderStep: invalid layer configuration");
    }

    const normalized = rmsNorm(input, layer.inputLayernorm, config.rmsNormEps);
    const q = splitHeads(linear(normalized, layer.qProj, qHeads * headDim, hidden), qHeads, headDim);
    const k = splitHeads(linear(normalized, layer.kProj, kvHeads * headDim, hidden), kvHeads, headDim);
    const v = splitHeads(linear(normalized, layer.vProj, kvHeads * headDim, hidden), kvHeads, headDim);

    const queries = q.map(head => applyRopeHalfSplit(rmsNorm(head, layer.qNorm, config.rmsNormEps), position, theta));
    const keys = k.map(head => applyRopeHalfSplit(rmsNorm(head, layer.kNorm, config.rmsNormEps), position, theta));

    // A cache entry is a list of KV heads, not a single flattened head.
    cache.append(layerIndex, position, keys, v, layerType, config.slidingWindow);
    const state = cache.layers[layerIndex];
    const attentionHeads = queries.map((query, qIndex) => {
        const kvIndex = Math.floor(qIndex / (qHeads / kvHeads));
        const keyHeads = state.keys.map(entry => entry[kvIndex]);
        const valueHeads = state.values.map(entry => entry[kvIndex]);

        return attentionForHead(
            query,
            keyHeads,
            valueHeads,
            state.positions,
            position,
            layerType,
            config.slidingWindow,
            config.queryPreAttnScalar
        );
    });

    const attention = linear(
        Float32Array.from(attentionHeads.flatMap(head => Array.from(head))),
        layer.oProj,
        hidden,
        qHeads * headDim
    );
    const afterAttention = add(input, rmsNorm(attention, layer.postAttentionLayernorm, config.rmsNormEps));
    const ffInput = rmsNorm(afterAttention, layer.preFeedforwardLayernorm, config.rmsNormEps);
    const gate = linear(ffInput, layer.gateProj, intermediate, hidden);
    const up = linear(ffInput, layer.upProj, intermediate, hidden);
    const activated = Float32Array.from(gate, (value, i) => geluTanh(value) * up[i]);
    const down = linear(activated, layer.downProj, hidden, intermediate);

    return add(afterAttention, rmsNorm(down, layer.postFeedforwardLayernorm, config.rmsNormEps));
}

export class DecoderSession {
    constructor(config, layers) {
        if (layers.length !== config.numLayers) {
            throw new Error("DecoderSession: layer count mismatch");
        }

        this.config = config;
        this.layers = layers;
        this.cache = new SessionCache(config.numLayers);
        this.position = 0;
    }

    step(embedding) {
        let hidden = Float32Array.from(embedding);

        for (let i = 0; i < this.layers.length; i += 1) {
            hidden = decoderStep(hidden, this.layers[i], this.cache, i, this.position, this.config);
        }

        this.position += 1;
        return hidden;
    }

    reset() {
        this.cache.clear();
        this.position = 0;
    }
}
