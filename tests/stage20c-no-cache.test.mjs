import assert from 'node:assert/strict';
import { Gemma3ConversationEngine } from '../src/gemma3-conversation-engine.js';

class ReferenceSession {
    constructor() {
        this.position = 0;
        this.history = [];
        this.replays = [];
        this.layers = [];
    }

    reset() {
        this.position = 0;
        this.history = [];
    }

    async step(value) {
        this.history.push(value);
        this.position += 1;
        return { output: this.history.reduce((a, b) => a * 3 + b, 0) };
    }

    async forwardNoCache(values) {
        this.reset();
        this.replays.push([...values]);
        let result;
        for (const value of values) {
            result = await this.step(value);
        }
        this.reset();
        return result;
    }
}

const session = new ReferenceSession();
const engine = new Gemma3ConversationEngine({
    session,
    tokenizer: {},
    embed: async id => id,
    predict: async () => ({ bestId: 1 }),
    eosTokenIds: [],
    useCache: false
});

const prefill = await engine.prefill([2, 3, 4]);
assert.equal(prefill.reused, 0);
assert.equal(prefill.processed, 3);
assert.equal(prefill.last.output, 2 * 9 + 3 * 3 + 4);
assert.deepEqual(session.replays, [[2], [2, 3], [2, 3, 4]]);
assert.equal(session.position, 0);

const next = await engine.stepToken(5);
assert.equal(next.output, 2 * 27 + 3 * 9 + 4 * 3 + 5);
assert.deepEqual(session.replays.at(-1), [2, 3, 4, 5]);
assert.equal(session.position, 0);

engine.setUseCache(true);
assert.equal(engine.cachedTokenIds.length, 0);
await engine.prefill([2, 3, 4]);
assert.equal(session.position, 3);
console.log('Stage 20C no-cache recompute policy PASS (mock; not GPU numerical parity)');
