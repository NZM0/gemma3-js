import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeInt4 } from '../src/backends/webgpu/weight-precision.js';
import { createChunkedHeadSource } from '../src/model-source.js';

test('INT4 signed nibble and group scale', () => {
    const words = new Uint32Array(10);
    const scales = new Float32Array(words.buffer);
    scales[0] = 0.5;
    scales[5] = 2;
    words.fill(0xfedcba98, 1, 5);
    words.fill(0x01234567, 6, 10);
    const result = decodeInt4(words);
    assert.deepEqual(Array.from(result.slice(0, 8)), [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]);
    assert.deepEqual(Array.from(result.slice(32, 40)), [-2, -4, -6, -8, -10, -12, -14, -16]);
});

test('aligned mobile head slice reuses fetched buffer', async () => {
    const data = new ArrayBuffer(20);
    const source = createChunkedHeadSource(name => {
        assert.equal(name, 'head-part-001.q4');
        return { arrayBuffer: async () => data };
    }, 20, 'q4');
    assert.equal(await source.slice(20, 40).arrayBuffer(), data);
});
