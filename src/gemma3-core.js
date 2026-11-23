/** Gemma 3 1B IT mathematical primitives. No model weights are bundled. */
export const GEMMA3_1B = Object.freeze({
    vocabSize: 262144,
    hiddenSize: 1152,
    intermediateSize: 6912,
    numLayers: 26,
    numHeads: 4,
    numKeyValueHeads: 1,
    headDim: 256,
    slidingWindow: 512,
    maxPositions: 32768,
    rmsNormEps: 1e-6,
    queryPreAttnScalar: 256,
    localRopeTheta: 10000,
    globalRopeTheta: 1000000,
    eosTokenIds: [1, 106],
    layerTypes: Array.from({ length: 26 }, (_, i) => (i % 6 === 5 ? "global" : "local"))
});

export function rmsNorm(values, weights, eps = 1e-6) {
    if (values.length !== weights.length) {
        throw new Error("RMSNorm: incompatible dimensions");
    }

    let sumSquares = 0;

    for (let i = 0; i < values.length; i += 1) {
        sumSquares += values[i] * values[i];
    }

    const inverseRms = 1 / Math.sqrt(sumSquares / values.length + eps);
    const output = new Float32Array(values.length);

    for (let i = 0; i < values.length; i += 1) {
        // Gemma checkpoints store a delta to 1, not the final multiplier.
        output[i] = values[i] * inverseRms * (1 + weights[i]);
    }

    return output;
}

export function geluTanh(x) {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

export function applyRopeHalfSplit(vector, position, theta = 10000) {
    if (vector.length % 2 !== 0) {
        throw new Error("RoPE requires an even head dimension");
    }

    const half = vector.length / 2;
    const output = new Float32Array(vector.length);

    for (let i = 0; i < half; i += 1) {
        const angle = position / theta ** (2 * i / vector.length);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const x = vector[i];
        const y = vector[i + half];

        output[i] = x * cosine - y * sine;
        output[i + half] = y * cosine + x * sine;
    }

    return output;
}

export function softmax(scores) {
    const maximum = Math.max(...scores);
    const exponentials = scores.map(value => Math.exp(value - maximum));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);

    return exponentials.map(value => value / denominator);
}

export function visibleKeyPositions(queryPosition, layerType, windowSize = 512) {
    const earliest = layerType === "local" ? Math.max(0, queryPosition - windowSize + 1) : 0;
    const positions = [];

    for (let position = earliest; position <= queryPosition; position += 1) {
        positions.push(position);
    }

    return positions;
}

export function attentionForHead(query, keys, values, keyPositions, queryPosition, layerType, windowSize = 512, scalar = 256) {
    const allowed = new Set(visibleKeyPositions(queryPosition, layerType, windowSize));
    const selected = [];
    const scores = [];

    for (let i = 0; i < keyPositions.length; i += 1) {
        if (!allowed.has(keyPositions[i])) {
            continue;
        }

        let dot = 0;

        for (let j = 0; j < query.length; j += 1) {
            dot += query[j] * keys[i][j];
        }

        selected.push(i);
        scores.push(dot / Math.sqrt(scalar));
    }

    if (scores.length === 0) {
        throw new Error("Attention has no visible keys");
    }

    const probabilities = softmax(scores);
    const output = new Float32Array(values[0].length);

    for (let i = 0; i < selected.length; i += 1) {
        const value = values[selected[i]];

        for (let j = 0; j < output.length; j += 1) {
            output[j] += probabilities[i] * value[j];
        }
    }

    return output;
}

/** Each character has its own cache; no sharing of private prompt state. */
export class SessionCache {
    constructor(numLayers = 26) {
        this.layers = Array.from({ length: numLayers }, () => ({ positions: [], keys: [], values: [] }));
    }

    append(layerIndex, position, key, value, layerType, windowSize = 512) {
        const layer = this.layers[layerIndex];
        layer.positions.push(position);
        layer.keys.push(key);
        layer.values.push(value);

        if (layerType === "local") {
            while (layer.positions.length > windowSize) {
                layer.positions.shift();
                layer.keys.shift();
                layer.values.shift();
            }
        }
    }

    clear() {
        for (const layer of this.layers) {
            layer.positions.length = 0;
            layer.keys.length = 0;
            layer.values.length = 0;
        }
    }
}
