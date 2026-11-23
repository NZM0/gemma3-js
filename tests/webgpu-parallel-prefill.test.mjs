import test from 'node:test';
import assert from 'node:assert/strict';
import { WebGpuParallelPrefill } from '../src/backends/webgpu/webgpu-parallel-prefill.js';

const previous = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 };

function mockDevice() {
    const events = [];
    const device = {
        limits: { maxBufferSize: 1 << 27, maxStorageBufferBindingSize: 1 << 27 },
        queue: { writeBuffer() {} },
        createShaderModule({ code }) { events.push(['shader', code]); return {}; },
        createComputePipeline({ compute }) {
            return { entry: compute.entryPoint, getBindGroupLayout() { return {}; } };
        },
        createBuffer({ size }) { return { size, destroy() {} }; },
        createBindGroup() { return {}; }
    };
    const encoder = {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
            assert.ok(srcOffset + size <= src.size);
            assert.ok(dstOffset + size <= dst.size);
            events.push(['copy', srcOffset, dstOffset, size]);
        },
        beginComputePass() {
            return {
                setPipeline(pipeline) { events.push(['kernel', pipeline.entry]); },
                setBindGroup() {},
                dispatchWorkgroups(n) { assert.ok(n > 0); },
                end() {}
            };
        }
    };
    return { device, encoder, events };
}

function layer(device, capacity = 512) {
    const d = 4;
    const h = 8;
    const f = 12;
    const make = n => device.createBuffer({ size: n * 4 });
    return {
        layerIndex: 0,
        maxCache: capacity,
        kv: make(capacity * d * 2),
        weights: Object.fromEntries([
            ['inputLayernorm', h], ['qProj', h * d], ['kProj', h * d],
            ['vProj', h * d], ['qNorm', d], ['kNorm', d],
            ['oProj', h * d], ['postAttentionLayernorm', h],
            ['preFeedforwardLayernorm', h], ['gateProj', f * h],
            ['upProj', f * h], ['downProj', h * f],
            ['postFeedforwardLayernorm', h]
        ].map(([name, n]) => [name, make(n)]))
    };
}

test('parallel prefill encodes token-wide matrix kernels and causal attention', () => {
    const { device, encoder, events } = mockDevice();
    const config = {
        hiddenSize: 8, headDim: 4, numHeads: 1, numKeyValueHeads: 1,
        intermediateSize: 12, slidingWindow: 512, layerTypes: ['local'],
        localRopeTheta: 10000, globalRopeTheta: 1000000,
        queryPreAttnScalar: 4, rmsNormEps: 1e-6
    };
    const runner = new WebGpuParallelPrefill(device, config);
    const temporary = [];
    const output = runner.encodeLayer(
        encoder, layer(device), device.createBuffer({ size: 8 * 4 * 4 }),
        0, 4, temporary
    );
    assert.equal(output.size, 8 * 4 * 4);
    assert.equal(events.filter(event => event[0] === 'kernel' && event[1] === 'attention').length, 1);
    assert.equal(events.filter(event => event[0] === 'kernel' && event[1] === 'matmul').length, 7);
    const source = events.find(event => event[0] === 'shader')[1];
    assert.match(source, /let stop = p\.history \+ t/);
    assert.match(source, /let start = max\(firstAbsolute/);
});

test('parallel prefill snapshots wrapped local ring with two contiguous copies', () => {
    const { device, encoder, events } = mockDevice();
    const config = {
        hiddenSize: 8, headDim: 4, numHeads: 1, numKeyValueHeads: 1,
        intermediateSize: 12, slidingWindow: 8, layerTypes: ['local'],
        localRopeTheta: 10000, globalRopeTheta: 1000000,
        queryPreAttnScalar: 4, rmsNormEps: 1e-6
    };
    const runner = new WebGpuParallelPrefill(device, config);
    runner.encodeLayer(encoder, layer(device, 8), device.createBuffer({ size: 8 * 2 * 4 }), 11, 2, []);
    const copies = events.filter(event => event[0] === 'copy');
    // Prior seven tokens are positions 4..10, crossing the ring boundary.
    assert.deepEqual(copies.slice(0, 2).map(event => event[3]), [4 * 32, 3 * 32]);
});

test.after(() => { globalThis.GPUBufferUsage = previous; });
