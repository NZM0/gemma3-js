import test from 'node:test';
import assert from 'node:assert/strict';
import { WebGpuFullModelSession } from '../src/backends/webgpu/webgpu-full-model.js';
import { WebGpuFullLayerSession } from '../src/backends/webgpu/webgpu-full-layer.js';
import { Gemma3ConversationEngine } from '../src/gemma3-conversation-engine.js';

globalThis.GPUBufferUsage = { COPY_DST: 1, COPY_SRC: 2, MAP_READ: 4 };
globalThis.GPUMapMode = { READ: 1 };

function fakeDevice() {
    const counters = { submissions: 0, readbacks: 0, copies: 0 };
    return {
        counters,
        createBuffer({ size }) {
            const bytes = new ArrayBuffer(size);
            return {
                destroy() {},
                async mapAsync() { counters.readbacks += 1; },
                getMappedRange() { return bytes; },
                unmap() {}
            };
        },
        createCommandEncoder() {
            return {
                copyBufferToBuffer() { counters.copies += 1; },
                finish() { return {}; }
            };
        },
        queue: {
            writeBuffer() {},
            submit() { counters.submissions += 1; }
        }
    };
}

function fakeLayer() {
    const layer = Object.create(WebGpuFullLayerSession.prototype);
    layer.position = 0;
    layer.layerIndex = 0;
    layer.maxCache = 512;
    layer.buffers = [];
    layer.calls = [];
    layer.prepareBatch = count => layer.calls.push(['reserve', count]);
    layer.encodeStep = (encoder, input, position, temporary, batch) => {
        assert.equal(position, layer.position);
        assert.equal(batch, true);
        layer.calls.push(['encode', position]);
        return input;
    };
    layer.reset = () => { layer.position = 0; };
    layer.dispose = () => {};
    return layer;
}

test('prefill submits once per chunk, preserves absolute positions and decode continuation', async () => {
    const device = fakeDevice();
    const layer = fakeLayer();
    const config = { numLayers: 1, hiddenSize: 4, layerTypes: ['local'], rmsNormEps: 1e-6 };
    const session = new WebGpuFullModelSession(device, [layer], new Float32Array(4), config);
    const embeddings = Array.from({ length: 5 }, () => new Float32Array([1, 2, 3, 4]));
    await session.prefill(embeddings, { batchSize: 2 });
    assert.equal(device.counters.submissions, 3);
    assert.equal(device.counters.readbacks, 3);
    assert.equal(session.position, 5);
    assert.equal(layer.position, 5);
    assert.deepEqual(layer.calls.filter(([type]) => type === 'encode').map(([, pos]) => pos), [0, 1, 2, 3, 4]);
    assert.deepEqual(layer.calls.filter(([type]) => type === 'reserve').map(([, n]) => n), [2, 2, 1]);
    session.dispose();
});

test('conversation uses batched prefill for cache-on and preserves exact token prefix', async () => {
    const calls = [];
    const session = {
        position: 0,
        layers: [],
        reset() { this.position = 0; },
        async prefill(embeddings) {
            calls.push(embeddings.length);
            this.position += embeddings.length;
            return { output: new Float32Array([1]) };
        }
    };
    const engine = new Gemma3ConversationEngine({
        session,
        tokenizer: {},
        embed: async id => new Float32Array([id]),
        predict: async () => ({}),
        eosTokenIds: []
    });
    await engine.prefill([1, 2, 3]);
    await engine.prefill([1, 2, 3, 4, 5]);
    assert.deepEqual(calls, [3, 2]);
    assert.deepEqual(engine.cachedTokenIds, [1, 2, 3, 4, 5]);
});
