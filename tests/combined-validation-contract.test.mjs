import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');

test('TF.js trace metadata identifies its actual backend', () => {
    const source = read('../app/tfjs/chat.js');
    assert.match(source, /backend: 'tfjs-webgl'/);
    assert.match(source, /gemma3-tfjs-webgl-trace\.json/);
});

test('both demos expose the same combined validation runner', () => {
    for (const backend of ['tfjs', 'webgpu']) {
        assert.match(read(`../app/${backend}/chat.js`), /runCombinedValidation\(/);
        assert.match(read(`../app/${backend}/index.html`), /id="validate-combined"/);
    }
});

test('TF.js cache-off path recomputes full prefix and resets cache', () => {
    const source = read('../src/backends/tfjs-webgl/tfjs-official-model.js');
    assert.match(source, /async forwardNoCache\(embeddings\)/);
    assert.match(source, /return await this\.forward\(flat\)/);
});


test('TF.js forward relies on tidy returned tensors, not redundant tf.keep', () => {
    const source = read('../src/backends/tfjs-webgl/tfjs-official-model.js');
    assert.doesNotMatch(source, /tf\.keep\(/);
    assert.match(source, /return \{ next, keys, values \}/);
});

test('combined validation compares multiple generated tokens and crosses 512 in prefill', () => {
    const source = read('../app/validation.js');
    assert.match(source, /run\(shortPrompt, true, 2\)/);
    assert.match(source, /run\(shortPrompt, false, 2\)/);
    assert.match(source, /tokenMatch/);
    assert.match(source, /tokenCount\(unit\.repeat\(upper\)\) < 520/);
    assert.doesNotMatch(source, /Could not construct a 507/);
});
