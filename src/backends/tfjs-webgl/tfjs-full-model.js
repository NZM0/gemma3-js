import { GEMMA3_1B, rmsNorm } from "../../gemma3-core.js";
import { TfjsLayerSession } from "./tfjs-layer.js";

/**
 * Complete 26-layer *hybrid* inference session. Dense projections execute on
 * TensorFlow.js WebGL, while attention, normalization and RoPE execute on CPU.
 * This is NOT an all-WebGL implementation or a performance-optimized backend.
 * A separate head/prediction implementation is required for token generation.
 */
export class TfjsFullModelSession {
    constructor(tf, layers, finalNorm, config = GEMMA3_1B) {
        if (layers.length !== config.numLayers) {
            throw new Error(`Expected ${config.numLayers} layers, got ${layers.length}`);
        }
        this.config = config;
        this.finalNorm = finalNorm;
        this.position = 0;
        this.disposed = false;
        this.layers = [];
        try {
            for (let index = 0; index < layers.length; index += 1) {
                this.layers.push(new TfjsLayerSession(tf, layers[index], config, index));
            }
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    async step(input, options = {}) {
        if (this.disposed) {
            throw new Error("Session disposed");
        }
        if (!(input instanceof Float32Array) || input.length !== this.config.hiddenSize) {
            throw new Error(`Expected Float32Array(${this.config.hiddenSize})`);
        }
        let hidden = input;
        const layerOutputs = [];
        for (const layer of this.layers) {
            hidden = await layer.step(hidden);
            if (options.trace === true) {
                layerOutputs.push(hidden.slice());
            }
        }
        this.position += 1;
        const lastNormalized = rmsNorm(hidden, this.finalNorm, this.config.rmsNormEps);
        return options.trace === true
            ? { output: hidden, lastNormalized, layerOutputs }
            : { output: hidden, lastNormalized };
    }

    async forwardNoCache(embeddings, options = {}) {
        if (!Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("forwardNoCache expects a nonempty array of embeddings");
        }
        this.reset();
        let result;
        try {
            for (const embedding of embeddings) {
                result = await this.step(embedding, options);
            }
            return result;
        } finally {
            this.reset();
        }
    }

    reset() {
        this.position = 0;
        for (const layer of this.layers) {
            layer.reset();
        }
    }

    dispose() {
        if (this.disposed) {
            return;
        }
        this.disposed = true;
        for (const layer of this.layers) {
            layer.dispose();
        }
        this.layers = [];
    }
}
