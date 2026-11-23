import { rmsNorm, GEMMA3_1B } from "./gemma3-core.js";
import { DecoderSession } from "./gemma3-decoder.js";

/** Greedy generation from a full, tied LM head. Float32 CPU diagnostic path. */
export class Gemma3Generator {
    constructor(layers, finalNorm, embeddingTable, config = GEMMA3_1B) {
        if (embeddingTable.length % config.hiddenSize !== 0) {
            throw new Error("Embedding table shape mismatch");
        }
        this.config = config;
        this.finalNorm = finalNorm;
        this.embeddingTable = embeddingTable;
        this.session = new DecoderSession(config, layers);
    }

    embedding(tokenId) {
        const hidden = this.config.hiddenSize;
        const vocab = this.embeddingTable.length / hidden;
        if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocab) {
            throw new Error(`Invalid token ID ${tokenId}`);
        }
        const result = new Float32Array(hidden);
        const scale = Math.sqrt(hidden);
        const offset = tokenId * hidden;
        for (let i = 0; i < hidden; i += 1) {
            result[i] = this.embeddingTable[offset + i] * scale;
        }
        return result;
    }

    nextLogits(hiddenState) {
        const normalized = rmsNorm(hiddenState, this.finalNorm, this.config.rmsNormEps);
        const hidden = this.config.hiddenSize;
        const vocab = this.embeddingTable.length / hidden;
        const logits = new Float32Array(vocab);
        for (let token = 0; token < vocab; token += 1) {
            let sum = 0;
            const offset = token * hidden;
            for (let i = 0; i < hidden; i += 1) {
                sum += normalized[i] * this.embeddingTable[offset + i];
            }
            logits[token] = sum;
        }
        return logits;
    }

    consume(tokenId) {
        return this.session.step(this.embedding(tokenId));
    }

    generate(promptIds, maxNewTokens = 8, onToken = null) {
        if (promptIds.length === 0) throw new Error("Prompt must not be empty");
        this.session.reset();
        let lastHidden;
        for (const id of promptIds) {
            lastHidden = this.consume(id);
        }
        const generated = [];
        for (let step = 0; step < maxNewTokens; step += 1) {
            const logits = this.nextLogits(lastHidden);
            let best = 0;
            for (let i = 1; i < logits.length; i += 1) {
                if (logits[i] > logits[best]) best = i;
            }
            generated.push(best);
            onToken?.(best, step, logits[best]);
            if (this.config.eosTokenIds.includes(best)) break;
            if (step + 1 < maxNewTokens) lastHidden = this.consume(best);
        }
        return generated;
    }
}
