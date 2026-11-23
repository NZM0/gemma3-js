import { DecoderSession } from "./gemma3-decoder.js";

function compareArrays(a, b) {
    if (a.length !== b.length) {
        throw new Error(`Array length mismatch: ${a.length} vs ${b.length}`);
    }

    let maximum = 0;
    let sumSquares = 0;

    for (let i = 0; i < a.length; i += 1) {
        const difference = Math.abs(a[i] - b[i]);

        if (!Number.isFinite(difference)) {
            throw new Error(`Non-finite comparison at index ${i}`);
        }

        maximum = Math.max(maximum, difference);
        sumSquares += difference * difference;
    }

    return {
        maxAbsoluteError: maximum,
        rmsError: Math.sqrt(sumSquares / a.length)
    };
}

function argmax(values) {
    let best = 0;

    for (let i = 1; i < values.length; i += 1) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}

/**
 * Compare two autoregressive decoding strategies using the same weights:
 *  - cached: prefill once, then reuse KV states across generated tokens;
 *  - recompute: start a fresh decoder session and replay the entire prefix
 *    for EVERY generated token. No KV state persists between generations.
 *
 * A decoder still needs the keys/values of earlier prefix positions while
 * computing an individual full-prefix forward pass. This test deliberately
 * disables cross-step KV reuse, not the attention mechanism itself.
 */
export function compareCachedAndRecomputed(generator, promptIds, maxNewTokens = 3, tolerance = 1e-5) {
    if (!Array.isArray(promptIds) || promptIds.length === 0) {
        throw new Error("Prompt must contain at least one token");
    }

    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1) {
        throw new Error("maxNewTokens must be a positive integer");
    }

    const cached = new DecoderSession(generator.config, generator.session.layers);
    const prefix = [...promptIds];
    const steps = [];
    const generatedTokenIds = [];
    let cachedHidden;

    for (const tokenId of promptIds) {
        cachedHidden = cached.step(generator.embedding(tokenId));
    }

    for (let step = 0; step < maxNewTokens; step += 1) {
        const recomputed = new DecoderSession(generator.config, generator.session.layers);
        let recomputedHidden;

        for (const tokenId of prefix) {
            recomputedHidden = recomputed.step(generator.embedding(tokenId));
        }

        const hiddenError = compareArrays(cachedHidden, recomputedHidden);
        const cachedLogits = generator.nextLogits(cachedHidden);
        const recomputedLogits = generator.nextLogits(recomputedHidden);
        const logitError = compareArrays(cachedLogits, recomputedLogits);
        const cachedTokenId = argmax(cachedLogits);
        const recomputedTokenId = argmax(recomputedLogits);
        const matches = cachedTokenId === recomputedTokenId
            && hiddenError.maxAbsoluteError <= tolerance
            && logitError.maxAbsoluteError <= tolerance;

        steps.push({
            step: step + 1,
            prefixLength: prefix.length,
            cachedTokenId,
            recomputedTokenId,
            hiddenError,
            logitError,
            matches
        });

        generatedTokenIds.push(cachedTokenId);

        if (!matches) {
            break;
        }

        if (generator.config.eosTokenIds.includes(cachedTokenId)) {
            break;
        }

        if (step + 1 < maxNewTokens) {
            prefix.push(cachedTokenId);
            cachedHidden = cached.step(generator.embedding(cachedTokenId));
        }
    }

    return {
        promptLength: promptIds.length,
        generatedTokenIds,
        steps,
        matches: steps.every(item => item.matches)
    };
}
