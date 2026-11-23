import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Gemma3ConversationEngine } from '../src/gemma3-conversation-engine.js';

const root = new URL('../', import.meta.url);
const source = path => readFileSync(new URL(path, root), 'utf8');

test('GPU final hidden state reaches LM head without CPU Float32 roundtrip', () => {
    const model = source('src/backends/webgpu/webgpu-full-model.js');
    const head = source('src/backends/webgpu/webgpu-lm-head.js');
    const app = source('app/webgpu/chat.js');
    assert.match(model, /gpuHiddenBuffer: this\.gpuOutput/);
    assert.match(head, /async predictResidentBuffer\(gpuHiddenBuffer\)/);
    assert.match(app, /session\.enableGpuOutput\(\)/);
    assert.match(app, /state\.head\.predictResidentBuffer\(hidden\)/);
});

test('timing is recorded on completed turn and first-token prediction uses GPU buffer', async () => {
    const buffer = { gpu: true };
    let position = 0;
    const session = {
        get position() { return position; },
        reset() { position = 0; },
        async prefill(inputs) { position += inputs.length; return { gpuHiddenBuffer: buffer }; },
        async step() { position += 1; return { gpuHiddenBuffer: buffer }; }
    };
    const seen = [];
    const engine = new Gemma3ConversationEngine({
        session,
        tokenizer: {
            encodeMessages: () => [2, 3],
            decode: ids => ids.join(',')
        },
        embed: async () => new Float32Array([1]),
        predict: async hidden => {
            seen.push(hidden);
            return { bestId: seen.length === 1 ? 7 : 106, bestLogit: 12 };
        },
        eosTokenIds: [106]
    });
    const result = await engine.chat('hello');
    assert.deepEqual(seen, [buffer, buffer]);
    assert.equal(result.generatedTokenIds.length, 2);
    assert.ok(Number.isFinite(result.timing.prefillMs));
    assert.ok(Number.isFinite(result.timing.ttftMs));
    assert.ok(Number.isFinite(result.timing.decodeTokensPerSecond));
});
