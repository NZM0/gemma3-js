/**
 * Text-only conversation controller for the independent Gemma 3 inference engine.
 *
 * The backend owns GPU K/V tensors. This controller owns their exact token prefix.
 * Reuse is allowed only when the next prompt begins with that prefix; otherwise
 * it resets and prefills from the beginning (never silently reuses stale K/V).
 *
 * This is a high-level cache policy, not a claim that Google exposes a `use_cache`
 * parameter with this name. Cache semantics must remain consistent with the
 * underlying Google Gemma 3 attention implementation.
 */
// Only backend-owned Tensor handles expose dispose(). WebGPU's gpuHiddenBuffer
// belongs to the session and must never be destroyed by this controller.
function disposeTransientResult(result) {
    const seen = new Set();
    for (const value of [result?.output, result?.lastNormalized]) {
        if (value && typeof value.dispose === 'function' && !seen.has(value)) {
            seen.add(value);
            if (!value.isDisposedInternal) { value.dispose(); }
        }
    }
}

export class Gemma3ConversationEngine {
    constructor({ session, tokenizer, embed, predict, eosTokenIds, useCache = true, maxContextTokens = 4096 }) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.embed = embed;
        this.predict = predict;
        this.eosTokenIds = new Set(eosTokenIds);
        this.useCache = useCache;
        // The effective limit must not exceed the GPU global-attention cache.
        // Local layers may use a smaller sliding-window ring buffer.
        const globalCapacities = session.layers
            ?.filter(layer => layer.config?.layerTypes?.[layer.layerIndex] === 'global')
            .map(layer => layer.maxCache) || [];
        const gpuLimit = globalCapacities.length > 0
            ? Math.min(...globalCapacities)
            : Infinity;
        this.maxContextTokens = Math.min(maxContextTokens, gpuLimit);
        this.lastPrefill = null;
        this.lastTiming = null;
        this.messages = [];
        this.cachedTokenIds = [];
    }

    reset() {
        this.session.reset();
        this.messages = [];
        this.cachedTokenIds = [];
        this.lastPrefill = null;
        this.lastTiming = null;
    }

    setUseCache(enabled) {
        this.useCache = Boolean(enabled);
        this.session.reset();
        this.cachedTokenIds = [];
        this.lastPrefill = null;
        this.lastTiming = null;
    }

    async stepToken(tokenId) {
        let result;
        if (this.useCache) {
            result = await this.session.step(await this.embed(tokenId));
        } else if (typeof this.session.forwardNoCache === "function") {
            // The full prefix is recomputed; no previous K/V is reused.
            const ids = [...this.cachedTokenIds, tokenId];
            const embeddings = [];
            for (const id of ids) {
                embeddings.push(await this.embed(id));
            }
            result = await this.session.forwardNoCache(embeddings);
        } else {
            // Preserve compatibility with mock/legacy sessions. Their
            // cache-off setting only disables cross-turn prefix reuse.
            result = await this.session.step(await this.embed(tokenId));
        }
        this.cachedTokenIds.push(tokenId);
        return result;
    }

    async prefill(tokenIds) {
        if (tokenIds.length === 0 || tokenIds.length > this.maxContextTokens) {
            throw new Error(`Invalid context length: ${tokenIds.length}`);
        }

        const matches = this.useCache &&
            this.cachedTokenIds.length < tokenIds.length &&
            this.cachedTokenIds.every((id, index) => id === tokenIds[index]) &&
            this.session.position === this.cachedTokenIds.length;

        if (!matches) {
            this.session.reset();
            this.cachedTokenIds = [];
        }

        const reused = this.cachedTokenIds.length;
        let last = null;
        if (this.useCache && typeof this.session.prefill === "function") {
            const missing = tokenIds.slice(reused);
            const embeddings = [];
            for (const id of missing) {
                embeddings.push(await this.embed(id));
            }
            last = await this.session.prefill(embeddings);
            this.cachedTokenIds.push(...missing);
        } else {
            for (let index = reused; index < tokenIds.length; index += 1) {
                const previous = last;
                try {
                    last = await this.stepToken(tokenIds[index]);
                } finally {
                    disposeTransientResult(previous);
                }
            }
        }

        const expectedPosition = !this.useCache && typeof this.session.forwardNoCache === "function"
            ? 0
            : this.cachedTokenIds.length;
        if (this.session.position !== expectedPosition) {
            throw new Error('GPU execution position diverged from expected cache mode');
        }
        this.lastPrefill = {
            total: tokenIds.length,
            reused,
            processed: tokenIds.length - reused
        };
        return { last, ...this.lastPrefill };
    }

    async chat(prompt, { maxNewTokens = 24, onToken = null, onPrefill = null } = {}) {
        if (typeof prompt !== 'string' || !prompt.trim()) {
            throw new Error('Prompt must be nonempty text');
        }
        if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1) {
            throw new Error('maxNewTokens must be a positive integer');
        }

        const nextMessages = [...this.messages, { role: 'user', content: prompt }];
        const ids = this.tokenizer.encodeMessages(nextMessages);
        let generated = [];
        const timing = {
            prefillMs: 0,
            headMs: 0,
            generatedStepMs: 0,
            tokens: 0,
            ttftMs: 0,
            decodeTokensPerSecond: 0,
            totalMs: 0
        };
        const requestStart = performance.now();

        try {
            const prefillStart = performance.now();
            const { last: initial, reused, processed } = await this.prefill(ids);
            timing.prefillMs = performance.now() - prefillStart;
            onPrefill?.({ total: ids.length, reused, processed, elapsedMs: timing.prefillMs });
            let last = initial;

            for (let index = 0; index < maxNewTokens; index += 1) {
                const headStart = performance.now();
                const { bestId, bestLogit } = await this.predict(last.gpuHiddenBuffer ?? last.output);
                timing.headMs += performance.now() - headStart;
                if (index === 0) { timing.ttftMs = performance.now() - requestStart; }
                timing.tokens += 1;
                generated.push(bestId);

                // Every generated token, including the final EOS or max-length
                // token, must enter the KV cache before a later turn can reuse it.
                // This also makes session.position equal cachedTokenIds.length.
                const finished = this.eosTokenIds.has(bestId);
                let text = null;
                try {
                    text = this.tokenizer.decode(generated);
                } catch (error) {
                    // Byte-fallback tokens can temporarily end mid-UTF-8.
                    if (finished || index + 1 === maxNewTokens) { throw error; }
                }
                onToken?.({ tokenId: bestId, logit: bestLogit, text, finished });

                if (this.cachedTokenIds.length >= this.maxContextTokens) {
                    if (!finished) { throw new Error('Context capacity reached during generation'); }
                    // EOS may be represented in history without materializing its K/V.
                    break;
                }
                const stepStart = performance.now();
                last = await this.stepToken(bestId);
                timing.generatedStepMs += performance.now() - stepStart;
                if (finished) { break; }
            }

            // The last generated token is stepped into KV for the next turn,
            // but its resulting hidden Tensor is not consumed by LM head.
            disposeTransientResult(last);
            const text = this.tokenizer.decode(generated);
            this.messages = [...nextMessages, { role: 'assistant', content: text }];
            timing.totalMs = performance.now() - requestStart;
            // Subsequent token latency includes both the previous GPU step and
            // the next LM-head prediction; exclude the first-token prediction.
            const decodeMs = timing.totalMs - timing.ttftMs;
            timing.decodeTokensPerSecond = generated.length > 1 && decodeMs > 0
                ? (generated.length - 1) * 1000 / decodeMs
                : 0;
            this.lastTiming = timing;
            return { text, generatedTokenIds: generated, history: this.messages.slice(), timing };
        } catch (error) {
            // No partial response or partially written cache may leak into
            // the next turn after an error.
            this.session.reset();
            this.cachedTokenIds = [];
            throw error;
        }
    }
}
