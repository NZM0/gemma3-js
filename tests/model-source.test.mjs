import test from 'node:test';
import assert from 'node:assert/strict';
import { createLocalModelSource, createRemoteModelSource } from '../src/model-source.js';

test('local source preserves folder paths and slices', async () => {
    const file = Object.assign(new Blob(['abcd']), { webkitRelativePath: 'model/layer/a' });
    const get = createLocalModelSource([file]);
    assert.equal(await get('layer/a').slice(1, 3).text(), 'bc');
    assert.throws(() => get('missing'), /Missing/);
});

test('remote source requests exact byte range', async () => {
    const get = createRemoteModelSource('https://example.com/model/', async (url, options) => {
        assert.equal(url, 'https://example.com/model/layer/a');
        assert.equal(options.headers.Range, 'bytes=1-2');
        return new Response('bc', { status: 206, headers: { 'Content-Range': 'bytes 1-2/4' } });
    });
    assert.equal(new TextDecoder().decode(await get('layer/a').slice(1, 3).arrayBuffer()), 'bc');
});

test('remote source rejects ignored ranges rather than buffering full file', async () => {
    const get = createRemoteModelSource('https://example.com/', async () => new Response('abcd'));
    await assert.rejects(get('a').slice(1, 3).arrayBuffer(), /exact byte ranges/);
});
