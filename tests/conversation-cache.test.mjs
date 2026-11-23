import assert from 'node:assert/strict';
import { Gemma3ConversationEngine } from '../src/gemma3-conversation-engine.js';

// A deterministic mock validates cache lifecycle and exact prefix handling.
// It does not claim numerical parity with the official PyTorch model.
class MockSession {
    constructor() {
        this.position = 0;
        this.tokens = [];
        this.resetCount = 0;
    }

    reset() {
        this.position = 0;
        this.tokens = [];
        this.resetCount += 1;
    }

    async step(token) {
        this.tokens.push(token);
        this.position += 1;
        return { output: this.tokens.reduce((a, b) => a * 17 + b, 0) };
    }
}

const tokenizer = {
    encodeMessages(messages) {
        const ids = [2];
        for (const message of messages) {
            ids.push(message.role === 'user' ? 10 : 11);
            ids.push(...Array.from(message.content, ch => ch.charCodeAt(0)));
            ids.push(106);
        }
        ids.push(11);
        return ids;
    },
    decode(ids) {
        return String.fromCharCode(...ids.filter(id => id !== 106));
    }
};

async function run(useCache) {
    const session = new MockSession();
    const engine = new Gemma3ConversationEngine({
        session,
        tokenizer,
        embed: async id => id,
        predict: async () => ({ bestId: 106, bestLogit: 1 }),
        eosTokenIds: [106],
        useCache
    });
    const first = await engine.chat('A');
    const firstPosition = session.position;
    const second = await engine.chat('B');
    const secondPrefill = { ...engine.lastPrefill };
    assert.deepEqual(first.generatedTokenIds, [106]);
    assert.deepEqual(second.generatedTokenIds, [106]);
    assert.equal(session.position, engine.cachedTokenIds.length);
    return { engine, session, firstPosition, secondPrefill };
}

const cached = await run(true);
const uncached = await run(false);
assert.ok(cached.secondPrefill.reused > 0);
assert.equal(uncached.secondPrefill.reused, 0);
assert.equal(cached.secondPrefill.total, uncached.secondPrefill.total);
assert.equal(cached.secondPrefill.processed + cached.secondPrefill.reused, cached.secondPrefill.total);

// Changing earlier history must never reuse an invalid suffix.
const modified = cached.engine.tokenizer.encodeMessages([
    { role: 'user', content: 'X' },
    { role: 'assistant', content: '' },
    { role: 'user', content: 'B' }
]);
const result = await cached.engine.prefill(modified);
assert.equal(result.reused, 0);

// Reset must clear both the logical prefix and the GPU-side position.
cached.engine.reset();
assert.equal(cached.session.position, 0);
assert.deepEqual(cached.engine.cachedTokenIds, []);
assert.equal(cached.engine.lastPrefill, null);

console.log('Conversation cache lifecycle PASS');
