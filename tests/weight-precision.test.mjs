import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeHalfArray, encodeHalfArray } from '../src/backends/webgpu/weight-precision.js';
import { createChunkedHeadSource } from '../src/model-source.js';

test('binary16 known values and rounding', () => {
    const bits = Uint16Array.of(0, 0x8000, 0x3c00, 0xc000, 1, 0x7bff);
    const values = decodeHalfArray(bits);
    assert.deepEqual(Array.from(values), [0, -0, 1, -2, 2 ** -24, 65504]);
    assert.deepEqual(encodeHalfArray(values), bits);
    assert.equal(encodeHalfArray([1 + 2 ** -11])[0], 0x3c00);
});

test('chunked head supports slices crossing part boundaries', async () => {
    const get = name => new Blob([name.includes('000') ? 'abcd' : 'efgh']);
    const head = createChunkedHeadSource(get, 4);
    assert.equal(new TextDecoder().decode(await head.slice(2, 6).arrayBuffer()), 'cdef');
});
