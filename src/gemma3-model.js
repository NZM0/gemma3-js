import { rmsNorm, GEMMA3_1B } from "./gemma3-core.js";
import { decoderStep, linear } from "./gemma3-decoder.js";
import { SessionCache } from "./gemma3-core.js";

/** Full 26-layer CPU reference. Each call is an independent sequence. */
export class Gemma3Model {
    constructor(layers, finalNorm, config = GEMMA3_1B) {
        if (layers.length !== config.numLayers) {
            throw new Error(`Expected ${config.numLayers} layers, received ${layers.length}`);
        }
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.config = config;
    }

    forwardEmbeddings(embeddings, onLayer = null) {
        const cache = new SessionCache(this.config.numLayers);
        const hiddenSize = this.config.hiddenSize;
        const length = embeddings.length / hiddenSize;
        if (!Number.isInteger(length)) {
            throw new Error("Embedding shape mismatch");
        }
        const outputs = new Float32Array(embeddings.length);
        for (let position = 0; position < length; position += 1) {
            let hidden = embeddings.slice(position * hiddenSize, (position + 1) * hiddenSize);
            for (let index = 0; index < this.layers.length; index += 1) {
                hidden = decoderStep(hidden, this.layers[index], cache, index, position, this.config);
                onLayer?.(index, position, hidden);
            }
            outputs.set(hidden, position * hiddenSize);
        }
        const last = outputs.subarray((length - 1) * hiddenSize);
        return { hidden: outputs, lastNormalized: rmsNorm(last, this.finalNorm, this.config.rmsNormEps) };
    }

    logitsForRows(lastNormalized, headRows) {
        return linear(lastNormalized, headRows, headRows.length / this.config.hiddenSize, this.config.hiddenSize);
    }
}
