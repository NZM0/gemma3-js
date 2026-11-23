import test from 'node:test';
import assert from 'node:assert/strict';
import { Gemma3ConversationEngine } from '../src/gemma3-conversation-engine.js';

function fixture(useCache) {
    const live = new Set();
    const session = {
        position: 0,
        layers: [],
        reset() { this.position = 0; },
        async step() {
            this.position += 1;
            const output = { dispose() { live.delete(output); }, isDisposedInternal: false };
            live.add(output);
            return { output, lastNormalized: output };
        },
        async forwardNoCache(embeddings) {
            this.position = 0;
            const output = { dispose() { live.delete(output); }, isDisposedInternal: false };
            live.add(output);
            return { output, lastNormalized: output };
        }
    };
    const engine = new Gemma3ConversationEngine({
        session,
        tokenizer: { encodeMessages() { return [1, 2, 3]; }, decode() { return 'ok'; } },
        embed: async () => new Float32Array(1),
        predict: async tensor => { tensor.dispose(); return { bestId: 7, bestLogit: 1 }; },
        eosTokenIds: [],
        useCache
    });
    return { engine, live };
}

for (const cache of [true, false]) {
    test(`transient hidden tensors are released after generation (cache=${cache})`, async () => {
        const { engine, live } = fixture(cache);
        await engine.chat('hi', { maxNewTokens: 2 });
        assert.equal(live.size, 0);
        engine.reset();
        assert.equal(live.size, 0);
    });
}
