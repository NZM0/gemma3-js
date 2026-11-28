import { decodeInt4, decodeHalfArray, encodeHalfArray } from "./weight-precision.js";
import { GEMMA3_1B } from "../../gemma3-core.js";

// Stage 15: GPU final RMSNorm and tied LM head, with chunked vocabulary projection.
// Full vocabulary mode reads logits back for CPU argmax (a future optimization).
const shader = `
struct Params { n: u32, m: u32, eps: f32, reserved: f32, };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> partial: array<f32, 64>;
fn sumGroup(value: f32, lane: u32) -> f32 {
    partial[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    let result = partial[0];
    workgroupBarrier();
    return result;
}

@compute @workgroup_size(64)
fn norm(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = 0u;
    var sum = 0.0;
    for (var j = lane; j < p.n; j += 64u) { sum += x[base + j] * x[base + j]; }
    let inv = inverseSqrt(sumGroup(sum, lane) / f32(p.n) + p.eps);
    for (var j = lane; j < p.n; j += 64u) {
        y[base + j] = x[base + j] * inv * (1.0 + w[j]);
    }
}
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if (r >= p.m) { return; }
    var acc = 0.0;
    for (var c = 0u; c < p.n; c = c + 1u) {
        acc += w[r * p.n + c] * x[c];
    }
    y[r] = acc;
}

// One workgroup per vocabulary chunk. Reduce chunk logits entirely on the GPU.
// The result is [bestLogit, bestTokenId] as f32; token IDs fit exactly in f32.
@group(0) @binding(4) var<storage, read_write> winners: array<vec2<f32>>;
var<workgroup> candidateLogits: array<f32, 64>;
var<workgroup> candidateIds: array<u32, 64>;
@compute @workgroup_size(64)
fn reduceChunk(@builtin(local_invocation_index) lane: u32) {
    var best = -3.402823e38;
    var bestId = 0u;
    for (var i = lane; i < p.m; i = i + 64u) {
        let value = x[i];
        if (value > best || (value == best && i < bestId)) {
            best = value;
            bestId = i;
        }
    }
    candidateLogits[lane] = best;
    candidateIds[lane] = bestId;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let other = candidateLogits[lane + stride];
            let otherId = candidateIds[lane + stride];
            if (other > candidateLogits[lane] || (other == candidateLogits[lane] && otherId < candidateIds[lane])) {
                candidateLogits[lane] = other;
                candidateIds[lane] = otherId;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        winners[0] = vec2<f32>(candidateLogits[0], f32(candidateIds[0]));
    }
}`;

const halfMatrixShader = `enable f16;
struct Params { n: u32, m: u32, eps: f32, reserved: f32, };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> halfWeights: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn matvecHalf(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if (r >= p.m) { return; }
    var acc = 0.0;
    for (var c = 0u; c < p.n; c = c + 1u) {
        acc += f32(halfWeights[r * p.n + c]) * x[c];
    }
    y[r] = acc;
}
`;

// Each 32-weight group occupies five u32 words: f32 scale, four packed words.
const int4MatrixShader = halfMatrixShader.replace('enable f16;', '')
    .replace('halfWeights: array<f16>', 'halfWeights: array<u32>')
    .replace('@compute @workgroup_size(64)', `fn weightAt(i: u32) -> f32 {
    let base = (i / 32u) * 5u;
    let lane = i % 32u;
    let packed = halfWeights[base + 1u + lane / 8u];
    let q = i32((packed >> ((lane % 8u) * 4u)) & 15u) - 8;
    return f32(q) * bitcast<f32>(halfWeights[base]);
}
@compute @workgroup_size(64)`)
    .replace(/f32\(halfWeights\[([^\]]+)\]\)/g, 'weightAt($1)')
    .replace(/for \(var [cj] = 0u;[^}]+\}/, `for (var g = 0u; g < p.n / 32u; g += 1u) {
        let offset = (r * p.n / 32u + g) * 5u;
        var subtotal = 0.0;
        for (var word = 0u; word < 4u; word += 1u) {
            let packed = halfWeights[offset + 1u + word];
            for (var lane = 0u; lane < 8u; lane += 1u) {
                let q = i32((packed >> (lane * 4u)) & 15u) - 8;
                subtotal += f32(q) * x[0u + g * 32u + word * 8u + lane];
            }
        }
        acc += subtotal * bitcast<f32>(halfWeights[offset]);
    }`);


// Eight lanes cooperate on each output row, eight rows per workgroup.
const parallelInt4Shader = int4MatrixShader.slice(0, int4MatrixShader.indexOf('@compute @workgroup_size(64)')) + `
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn matvecHalf(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.x * 8u + lane / 8u;
    let part = lane % 8u;
    var acc = 0.0;
    if (row < p.m) {
        for (var g = part; g < p.n / 32u; g += 8u) {
            let offset = (row * p.n / 32u + g) * 5u;
            var subtotal = 0.0;
            for (var word = 0u; word < 4u; word += 1u) {
                let packed = halfWeights[offset + 1u + word];
                for (var j = 0u; j < 8u; j += 1u) {
                    let q = i32((packed >> (j * 4u)) & 15u) - 8;
                    subtotal += f32(q) * x[g * 32u + word * 8u + j];
                }
            }
            acc += subtotal * bitcast<f32>(halfWeights[offset]);
        }
    }
    sums[lane] = acc;
    workgroupBarrier();
    for (var stride = 4u; stride > 0u; stride /= 2u) {
        if (part < stride) { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    if (part == 0u && row < p.m) { y[row] = sums[lane]; }
}
`;


export class WebGpuLmHead {
    constructor(device, normWeights, hiddenSize = GEMMA3_1B.hiddenSize, eps = GEMMA3_1B.rmsNormEps, halfWeights = false, int4Weights = false) {
        this.device = device;
        this.hiddenSize = hiddenSize;
        this.eps = eps;
        this.halfWeights = halfWeights;
        this.int4Weights = int4Weights;
        this.module = device.createShaderModule({ code: shader });
        this.pipelines = {};
        for (const name of ["norm", "matvec", "reduceChunk"]) {
            this.pipelines[name] = device.createComputePipeline({
                layout: "auto",
                compute: { module: this.module, entryPoint: name }
            });
        }
        if (halfWeights || int4Weights) {
            const module = device.createShaderModule({ code: int4Weights ? parallelInt4Shader : halfMatrixShader });
            this.pipelines.matvec = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "matvecHalf" } });
        }
        this.normWeight = this.upload(normWeights);
        this.input = this.make(hiddenSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.normalized = this.make(hiddenSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.buffers = [this.normWeight, this.input, this.normalized];
        this.residentChunks = [];
        this.residentVocab = 0;
    }

    make(size, usage) {
        return this.device.createBuffer({ size: Math.max(4, size), usage });
    }

    upload(values) {
        const buffer = this.make(values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(buffer, 0, values);
        return buffer;
    }

    dispatch(encoder, name, input, weight, output, n, m) {
        const params = new ArrayBuffer(16);
        const view = new DataView(params);
        view.setUint32(0, n, true);
        view.setUint32(4, m, true);
        view.setFloat32(8, this.eps, true);
        const uniform = this.make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(uniform, 0, params);
        const pipeline = this.pipelines[name];
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: input } },
                { binding: 1, resource: { buffer: weight } },
                { binding: 2, resource: { buffer: output } },
                { binding: 3, resource: { buffer: uniform } }
            ]
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(name === "norm" ? 1 : Math.ceil(m / (this.int4Weights ? 8 : 64)));
        pass.end();
        return uniform;
    }


    // Keep the tied embedding/LM-head matrix resident on the GPU after model load.
    // Each chunk is below maxStorageBufferBindingSize; no weight upload during decode.
    async loadResidentWeights(headFile, vocab, onProgress = () => {}) {
        this.clearResidentWeights();
        const rowBytes = this.hiddenSize * (this.int4Weights ? 20 / 32 : this.halfWeights ? 2 : 4);
        const maxRows = Math.max(1, Math.floor(this.device.limits.maxStorageBufferBindingSize / rowBytes));
        const chunkRows = Math.min(8192, maxRows);

        try {
            for (let start = 0; start < vocab; start += chunkRows) {
                const end = Math.min(vocab, start + chunkRows);
                const ArrayType = this.int4Weights ? Uint32Array : this.halfWeights ? Uint16Array : Float32Array;
                const rows = new ArrayType(await headFile.slice(start * rowBytes, end * rowBytes).arrayBuffer());
                if (rows.byteLength !== (end - start) * rowBytes) {
                    throw new Error(`LM-head file truncated at row ${start}`);
                }
                const weight = this.make(rows.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
                this.device.queue.writeBuffer(weight, 0, rows);
                this.residentChunks.push({ start, end, weight });
                await this.device.queue.onSubmittedWorkDone();
                onProgress(end, vocab);
            }
            this.residentVocab = vocab;
        } catch (error) {
            this.clearResidentWeights();
            throw error;
        }
    }

    async readEmbedding(tokenId) {
        const chunk = this.residentChunks.find(entry => tokenId >= entry.start && tokenId < entry.end);
        if (!chunk) { throw new Error(`Invalid embedding token: ${tokenId}`); }
        const bytes = this.hiddenSize * (this.int4Weights ? 20 / 32 : this.halfWeights ? 2 : 4);
        const buffer = this.make(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        try {
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(chunk.weight, (tokenId - chunk.start) * bytes, buffer, 0, bytes);
            this.device.queue.submit([encoder.finish()]);
            await buffer.mapAsync(GPUMapMode.READ);
            const data = buffer.getMappedRange().slice(0);
            if (this.int4Weights) { return decodeInt4(new Uint32Array(data)); }
            return this.halfWeights ? decodeHalfArray(new Uint16Array(data)) : new Float32Array(data);
        } finally {
            buffer.destroy();
        }
    }

    clearResidentWeights() {
        for (const chunk of this.residentChunks) {
            chunk.weight.destroy();
        }
        this.residentChunks = [];
        this.residentVocab = 0;
    }

    async predictResident(hidden) {
        if (hidden.length !== this.hiddenSize || this.residentVocab === 0) {
            throw new Error('Resident LM-head weights missing or hidden dimension mismatch');
        }
        this.device.queue.writeBuffer(this.input, 0, hidden);
        return this.predictResidentBuffer(this.input);
    }

    async predictResidentBuffer(gpuHiddenBuffer) {
        if (!gpuHiddenBuffer || this.residentVocab === 0) {
            throw new Error('GPU hidden buffer or resident LM-head weights missing');
        }
        const count = this.residentChunks.length;
        const results = this.make(count * 8, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const readback = this.make(count * 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = this.device.createCommandEncoder();
        const temporaries = [results, readback];

        try {
            temporaries.push(this.dispatch(encoder, 'norm', gpuHiddenBuffer, this.normWeight,
                this.normalized, this.hiddenSize, this.hiddenSize));

            for (let index = 0; index < count; index += 1) {
                const chunk = this.residentChunks[index];
                const rows = chunk.end - chunk.start;
                const output = this.make(rows * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
                const winner = this.make(8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
                temporaries.push(output, winner);
                temporaries.push(this.dispatch(encoder, 'matvec', this.normalized,
                    chunk.weight, output, this.hiddenSize, rows));

                const uniform = this.make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                temporaries.push(uniform);
                const params = new ArrayBuffer(16);
                const view = new DataView(params);
                view.setUint32(0, rows, true);
                view.setUint32(4, rows, true);
                this.device.queue.writeBuffer(uniform, 0, params);
                const pipeline = this.pipelines.reduceChunk;
                const group = this.device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: output } },
                        { binding: 4, resource: { buffer: winner } },
                        { binding: 3, resource: { buffer: uniform } }
                    ]
                });
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.dispatchWorkgroups(1);
                pass.end();
                encoder.copyBufferToBuffer(winner, 0, results, index * 8, 8);
            }
            encoder.copyBufferToBuffer(results, 0, readback, 0, count * 8);
            this.device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const winners = new Float32Array(readback.getMappedRange().slice(0));
            readback.unmap();

            let bestId = -1;
            let bestLogit = -Infinity;
            for (let index = 0; index < count; index += 1) {
                const value = winners[index * 2];
                const tokenId = this.residentChunks[index].start + winners[index * 2 + 1];
                if (!Number.isFinite(value)) {
                    throw new Error(`Non-finite LM-head logit in chunk ${index}`);
                }
                if (value > bestLogit || (value === bestLogit && tokenId < bestId)) {
                    bestLogit = value;
                    bestId = tokenId;
                }
            }
            return { bestId, bestLogit };
        } finally {
            for (const buffer of temporaries) { buffer?.destroy(); }
        }
    }

    async logits(hidden, rows) {
        if (this.int4Weights) { throw new Error("Use resident INT4 head for prediction"); }
        if (hidden.length !== this.hiddenSize || rows.length % this.hiddenSize !== 0) {
            throw new Error("LM head dimensions mismatch");
        }
        const count = rows.length / this.hiddenSize;
        const weight = this.upload(this.halfWeights && rows instanceof Float32Array ? encodeHalfArray(rows) : rows);
        const output = this.make(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const readback = this.make(count * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        this.device.queue.writeBuffer(this.input, 0, hidden);
        const encoder = this.device.createCommandEncoder();
        const u1 = this.dispatch(encoder, "norm", this.input, this.normWeight, this.normalized, this.hiddenSize, this.hiddenSize);
        const u2 = this.dispatch(encoder, "matvec", this.normalized, weight, output, this.hiddenSize, count);
        encoder.copyBufferToBuffer(output, 0, readback, 0, count * 4);
        try {
            this.device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(readback.getMappedRange().slice(0));
            readback.unmap();
            if (!result.every(Number.isFinite)) { throw new Error("Non-finite GPU LM-head logits"); }
            return result;
        } finally {
            for (const buffer of [weight, output, readback, u1, u2]) { buffer.destroy(); }
        }
    }

    dispose() {
        this.clearResidentWeights();
        for (const buffer of this.buffers) { buffer.destroy(); }
    }
}
