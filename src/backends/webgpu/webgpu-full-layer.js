import { kvWindowPlan } from "../../gemma3-kv-window-plan.js";
import { GEMMA3_1B } from "../../gemma3-core.js";

const wgsl = `
struct Params { n: u32, m: u32, position: u32, cacheLength: u32, theta: f32, scalar: f32, eps: f32, capacity: u32, };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

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
@compute @workgroup_size(64)
fn norm(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    var sq = 0.0;
    for (var j = 0u; j < p.n; j = j + 1u) {
        sq += x[j] * x[j];
    }
    y[i] = x[i] * inverseSqrt(sq / f32(p.n) + p.eps) * (1.0 + w[i % p.m]);
}
@compute @workgroup_size(64)
fn qknormrope(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    let dim = p.m;
    let half = dim / 2u;
    let base = (i / dim) * dim;
    let within = i % dim;
    var sq = 0.0;
    for (var j = 0u; j < dim; j = j + 1u) {
        sq += x[base + j] * x[base + j];
    }
    let inv = inverseSqrt(sq / f32(dim) + p.eps);
    let pair = within % half;
    let a = x[base + pair] * inv * (1.0 + w[pair]);
    let b = x[base + pair + half] * inv * (1.0 + w[pair + half]);
    let angle = f32(p.position) / pow(p.theta, 2.0 * f32(pair) / f32(dim));
    let cs = cos(angle);
    let sn = sin(angle);
    if (within < half) { y[i] = a * cs - b * sn; }
    else { y[i] = b * cs + a * sn; }
}
@compute @workgroup_size(64)
fn copyvec(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < p.n) { y[id.x] = x[id.x]; }
}
@compute @workgroup_size(64)
fn addnorm(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    var sq = 0.0;
    for (var j = 0u; j < p.n; j = j + 1u) { sq += x[j] * x[j]; }
    y[i] = y[i] + x[i] * inverseSqrt(sq / f32(p.n) + p.eps) * (1.0 + w[i]);
}
@compute @workgroup_size(64)
fn gelumul(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    let v = x[i];
    let t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    let g = 0.5 * v * (1.0 + tanh(clamp(t, -10.0, 10.0)));
    y[i] = g * w[i];
}
@compute @workgroup_size(64)
fn attn(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    let headDim = p.m;
    let head = i / headDim;
    let component = i % headDim;
    let keys = p.cacheLength;
    let keyStride = headDim * 2u;
    let capacity = p.capacity;
    let firstPosition = p.position + 1u - keys;
    var maxScore = -3.402823e38;
    for (var t = 0u; t < keys; t = t + 1u) {
        var dot = 0.0;
        for (var j = 0u; j < headDim; j = j + 1u) {
            dot += x[head * headDim + j] * w[((firstPosition + t) % capacity) * keyStride + j];
        }
        maxScore = max(maxScore, dot / sqrt(p.scalar));
    }
    var denominator = 0.0;
    var numerator = 0.0;
    for (var t = 0u; t < keys; t = t + 1u) {
        var dot = 0.0;
        for (var j = 0u; j < headDim; j = j + 1u) {
            dot += x[head * headDim + j] * w[((firstPosition + t) % capacity) * keyStride + j];
        }
        let prob = exp(dot / sqrt(p.scalar) - maxScore);
        denominator += prob;
        numerator += prob * w[((firstPosition + t) % capacity) * keyStride + headDim + component];
    }
    y[i] = numerator / denominator;
}
`;

/** Stage 13: GPU-resident one-token transformer block. No CPU intermediate readbacks. */
const halfMatrixShader = `enable f16;
struct Params { n: u32, m: u32, position: u32, cacheLength: u32, theta: f32, scalar: f32, eps: f32, capacity: u32, };
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
    .replace(/f32\(halfWeights\[([^\]]+)\]\)/g, 'weightAt($1)');


export class WebGpuFullLayerSession {
    constructor(device, layer, config = GEMMA3_1B, layerIndex = 0) {
        this.device = device;
        this.config = config;
        this.layerIndex = layerIndex;
        this.halfWeights = layer.qProj instanceof Uint16Array;
        this.int4Weights = layer.qProj instanceof Uint32Array;
        this.position = 0;
        this.disposed = false;
        this.buffers = [];
        this.weights = {};
        this.cacheKeys = [];
        this.cacheValues = [];
        this.maxCache = config.layerTypes[layerIndex] === "local" ? config.slidingWindow : config.maxPositions;
        // Never silently shrink the official attention window: doing so changes logits.
        const kvBytes = this.maxCache * config.headDim * 2 * Float32Array.BYTES_PER_ELEMENT;
        if (kvBytes > device.limits.maxStorageBufferBindingSize || kvBytes > device.limits.maxBufferSize) {
            throw new Error(`Layer ${layerIndex}: required ${this.maxCache}-token KV buffer (${kvBytes} bytes) exceeds WebGPU limits. A different storage layout is required; reducing the attention window is not valid.`);
        }
        this.shader = device.createShaderModule({ code: wgsl });
        this.pipelines = {};
        for (const entry of ["matvec", "norm", "qknormrope", "copyvec", "addnorm", "gelumul", "attn"]) {
            this.pipelines[entry] = device.createComputePipeline({ layout: "auto", compute: { module: this.shader, entryPoint: entry } });
        }
        if (this.halfWeights || this.int4Weights) {
            const module = device.createShaderModule({ code: this.int4Weights ? int4MatrixShader : halfMatrixShader });
            this.pipelines.matvec = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "matvecHalf" } });
        }
        for (const [name, data] of Object.entries(layer)) {
            if (!(data instanceof Float32Array || data instanceof Uint16Array || data instanceof Uint32Array)) { continue; }
            this.weights[name] = this.make(data.byteLength / 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
            device.queue.writeBuffer(this.weights[name], 0, data);
        }
        this.hidden = this.make(config.hiddenSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.normed = this.make(config.hiddenSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.q = this.make(config.numHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.k = this.make(config.numKeyValueHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.v = this.make(config.numKeyValueHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.qr = this.make(config.numHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.kr = this.make(config.numKeyValueHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.att = this.make(config.numHeads * config.headDim, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.projected = this.make(config.hiddenSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.ff = this.make(config.hiddenSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.gate = this.make(config.intermediateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.up = this.make(config.intermediateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.activated = this.make(config.intermediateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.down = this.make(config.hiddenSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.kv = this.make(this.maxCache * config.headDim * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.readback = this.make(config.hiddenSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        // 256-byte-aligned uniform records, reusable after each submitted step completes.
        this.uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment || 256);
        this.uniformCapacity = 32;
        this.uniformCursor = 0;
        this.uniformArena = this.device.createBuffer({
            size: this.uniformStride * this.uniformCapacity,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.buffers.push(this.uniformArena);
    }

    make(length, usage) {
        const bytes = Math.max(4, length * 4);
        if (bytes > this.device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && bytes > this.device.limits.maxStorageBufferBindingSize)) {
            throw new Error(`GPU buffer of ${bytes} bytes exceeds device limit`);
        }
        const buffer = this.device.createBuffer({ size: bytes, usage });
        this.buffers.push(buffer);
        return buffer;
    }

    dispatch(encoder, entry, x, w, y, n, m, position, cacheLength, theta) {
        const params = new ArrayBuffer(32);
        const view = new DataView(params);
        view.setUint32(0, n, true);
        view.setUint32(4, m, true);
        view.setUint32(8, position, true);
        view.setUint32(12, cacheLength, true);
        view.setFloat32(16, theta, true);
        view.setFloat32(20, this.config.queryPreAttnScalar, true);
        view.setFloat32(24, this.config.rmsNormEps, true);
        view.setUint32(28, this.maxCache, true);
        if (this.uniformCursor >= this.uniformCapacity) {
            throw new Error(`Layer ${this.layerIndex}: uniform arena exhausted`);
        }
        const uniformOffset = this.uniformCursor * this.uniformStride;
        this.uniformCursor += 1;
        this.device.queue.writeBuffer(this.uniformArena, uniformOffset, params);
        const pipeline = this.pipelines[entry];
        const group = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            // Auto pipeline layouts contain only bindings actually referenced by each entry point.
            // copyvec does not use binding 1 (w), so it must not be supplied.
            entries: (entry === "copyvec"
                ? [[0, x], [2, y], [3, this.uniformArena]]
                : [[0, x], [1, w], [2, y], [3, this.uniformArena]]
            ).map(([binding, buffer]) => ({
                binding,
                resource: binding === 3
                    ? { buffer, offset: uniformOffset, size: 32 }
                    : { buffer }
            }))
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(n === 0 ? 1 : (entry === "matvec" ? m : n) / 64));
        pass.end();
        return null;
    }

    /** Reserve distinct uniform records for a multi-token command submission.
     * Queue writes must not overwrite parameters of earlier encoded dispatches.
     * Call only after the preceding submission has completed.
     */
    prepareBatch(tokens) {
        if (!Number.isInteger(tokens) || tokens < 1) {
            throw new Error("Batch token count must be a positive integer");
        }
        const required = tokens * 32;
        if (required > this.uniformCapacity) {
            const bytes = this.uniformStride * required;
            if (bytes > this.device.limits.maxBufferSize) {
                throw new Error(`Layer ${this.layerIndex}: uniform batch exceeds GPU buffer limit`);
            }
            const previous = this.uniformArena;
            this.uniformArena = this.device.createBuffer({
                size: bytes,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.buffers = this.buffers.filter(buffer => buffer !== previous);
            previous.destroy();
            this.buffers.push(this.uniformArena);
            this.uniformCapacity = required;
        }
        this.uniformCursor = 0;
    }

    /** Encode one GPU-resident layer. Input and output remain GPUBuffer objects.
     *  The caller submits one command buffer for the entire 26-layer model.
     *  Do not reset position or reuse a layer before the previous submission finishes.
     */
    encodeStep(encoder, inputBuffer, position, temporary, batch = false) {
        if (this.disposed) { throw new Error("Session disposed"); }
        if (position !== this.position) {
            throw new Error(`Layer ${this.layerIndex}: expected position ${this.position}, received ${position}`);
        }
        if (!batch) { this.uniformCursor = 0; }
        const c = this.config;
        const h = c.hiddenSize;
        const d = c.headDim;
        const intermediate = c.intermediateSize;
        if (position >= this.maxCache && c.layerTypes[this.layerIndex] !== "local") {
            throw new Error(`Layer ${this.layerIndex}: global KV cache capacity exceeded`);
        }
        const theta = c.layerTypes[this.layerIndex] === "local" ? c.localRopeTheta : c.globalRopeTheta;
        const run = (entry, x, w, y, n, m, length = 0) => {
            this.dispatch(encoder, entry, x, w, y, n, m, position, length, theta);
        };

        // The preceding layer's output is already on the GPU.
        encoder.copyBufferToBuffer(inputBuffer, 0, this.hidden, 0, h * 4);
        run("norm", this.hidden, this.weights.inputLayernorm, this.normed, h, h);
        run("matvec", this.normed, this.weights.qProj, this.q, h, c.numHeads * d);
        run("matvec", this.normed, this.weights.kProj, this.k, h, c.numKeyValueHeads * d);
        run("matvec", this.normed, this.weights.vProj, this.v, h, c.numKeyValueHeads * d);
        run("qknormrope", this.q, this.weights.qNorm, this.qr, c.numHeads * d, d);
        run("qknormrope", this.k, this.weights.kNorm, this.kr, c.numKeyValueHeads * d, d);

        const plan = kvWindowPlan(position, this.maxCache);
        const slot = plan.writeSlot;
        encoder.copyBufferToBuffer(this.kr, 0, this.kv, slot * d * 8, d * 4);
        encoder.copyBufferToBuffer(this.v, 0, this.kv, slot * d * 8 + d * 4, d * 4);
        // The attention shader reads the circular KV buffer in chronological order.
        // No per-token ordered scratch buffer or GPU-to-GPU window copies are needed.
        const length = plan.length;
        run("attn", this.qr, this.kv, this.att, c.numHeads * d, d, length);
        run("matvec", this.att, this.weights.oProj, this.projected, c.numHeads * d, h);
        run("copyvec", this.hidden, this.weights.inputLayernorm, this.ff, h, h);
        run("addnorm", this.projected, this.weights.postAttentionLayernorm, this.ff, h, h);
        run("norm", this.ff, this.weights.preFeedforwardLayernorm, this.normed, h, h);
        run("matvec", this.normed, this.weights.gateProj, this.gate, h, intermediate);
        run("matvec", this.normed, this.weights.upProj, this.up, h, intermediate);
        run("gelumul", this.gate, this.up, this.activated, intermediate, intermediate);
        run("matvec", this.activated, this.weights.downProj, this.down, intermediate, h);
        run("addnorm", this.down, this.weights.postFeedforwardLayernorm, this.ff, h, h);
        return this.ff;
    }

    async step(input, options = {}) {
        const diagnostic = options.diagnostic === true;
        const snapshots = [];
        const capture = (encoder, name, buffer, length) => {
            if (!diagnostic) { return; }
            const staging = this.make(length, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            encoder.copyBufferToBuffer(buffer, 0, staging, 0, length * 4);
            snapshots.push({ name, staging });
        };
        if (this.disposed) { throw new Error("Session disposed"); }
        this.uniformCursor = 0;
        const c = this.config;
        const h = c.hiddenSize;
        const d = c.headDim;
        const intermediate = c.intermediateSize;
        if (input.length !== h) { throw new Error(`Expected ${h} hidden values`); }
        if (this.position >= this.maxCache && c.layerTypes[this.layerIndex] !== "local") {
            throw new Error("Global KV cache capacity exceeded");
        }
        const theta = c.layerTypes[this.layerIndex] === "local" ? c.localRopeTheta : c.globalRopeTheta;
        const pos = this.position;
        const uniforms = [];
        const run = (encoder, entry, x, w, y, n, m, length = 0) => {
            this.dispatch(encoder, entry, x, w, y, n, m, pos, length, theta);
        };
        this.device.queue.writeBuffer(this.hidden, 0, input);
        const encoder = this.device.createCommandEncoder();
        capture(encoder, "inputHidden", this.hidden, h);
        capture(encoder, "inputNormWeights", this.weights.inputLayernorm, h);
        run(encoder, "norm", this.hidden, this.weights.inputLayernorm, this.normed, h, h);
        capture(encoder, "inputNorm", this.normed, h);
        run(encoder, "matvec", this.normed, this.weights.qProj, this.q, h, c.numHeads * d);
        run(encoder, "matvec", this.normed, this.weights.kProj, this.k, h, c.numKeyValueHeads * d);
        run(encoder, "matvec", this.normed, this.weights.vProj, this.v, h, c.numKeyValueHeads * d);
        capture(encoder, "qProjection", this.q, c.numHeads * d);
        capture(encoder, "kProjection", this.k, c.numKeyValueHeads * d);
        capture(encoder, "vProjection", this.v, c.numKeyValueHeads * d);
        run(encoder, "qknormrope", this.q, this.weights.qNorm, this.qr, c.numHeads * d, d);
        run(encoder, "qknormrope", this.k, this.weights.kNorm, this.kr, c.numKeyValueHeads * d, d);
        capture(encoder, "qRope", this.qr, c.numHeads * d);
        capture(encoder, "kRope", this.kr, c.numKeyValueHeads * d);
        // GPU -> GPU copies; CPU never reads Q/K/V or hidden intermediates.
        const plan = kvWindowPlan(pos, this.maxCache);
        const slot = plan.writeSlot;
        encoder.copyBufferToBuffer(this.kr, 0, this.kv, slot * d * 8, d * 4);
        encoder.copyBufferToBuffer(this.v, 0, this.kv, slot * d * 8 + d * 4, d * 4);
        const length = plan.length;
        run(encoder, "attn", this.qr, this.kv, this.att, c.numHeads * d, d, length);
        capture(encoder, "attention", this.att, c.numHeads * d);
        run(encoder, "matvec", this.att, this.weights.oProj, this.projected, c.numHeads * d, h);
        capture(encoder, "oProjection", this.projected, h);
        run(encoder, "copyvec", this.hidden, this.weights.inputLayernorm, this.ff, h, h);
        run(encoder, "addnorm", this.projected, this.weights.postAttentionLayernorm, this.ff, h, h);
        capture(encoder, "afterAttention", this.ff, h);
        run(encoder, "norm", this.ff, this.weights.preFeedforwardLayernorm, this.normed, h, h);
        capture(encoder, "ffNorm", this.normed, h);
        run(encoder, "matvec", this.normed, this.weights.gateProj, this.gate, h, intermediate);
        run(encoder, "matvec", this.normed, this.weights.upProj, this.up, h, intermediate);
        capture(encoder, "gate", this.gate, intermediate);
        capture(encoder, "up", this.up, intermediate);
        run(encoder, "gelumul", this.gate, this.up, this.activated, intermediate, intermediate);
        capture(encoder, "activated", this.activated, intermediate);
        run(encoder, "matvec", this.activated, this.weights.downProj, this.down, intermediate, h);
        capture(encoder, "down", this.down, h);
        run(encoder, "addnorm", this.down, this.weights.postFeedforwardLayernorm, this.ff, h, h);
        capture(encoder, "output", this.ff, h);
        encoder.copyBufferToBuffer(this.ff, 0, this.readback, 0, h * 4);
        this.device.queue.submit([encoder.finish()]);
        try {
            await this.readback.mapAsync(GPUMapMode.READ);
            const output = new Float32Array(this.readback.getMappedRange().slice(0));
            this.readback.unmap();
            const trace = {};
            for (const { name, staging } of snapshots) {
                await staging.mapAsync(GPUMapMode.READ);
                trace[name] = new Float32Array(staging.getMappedRange().slice(0));
                staging.unmap();
            }
            this.position += 1;
            return diagnostic ? { output, trace } : output;
        } finally {
            const temporary = [...uniforms, ...snapshots.map(item => item.staging)];
            for (const buffer of temporary) { buffer.destroy(); }
            const temporarySet = new Set(temporary);
            this.buffers = this.buffers.filter(buffer => !temporarySet.has(buffer));
        }
    }

    reset() {
        this.position = 0;
    }

    dispose() {
        if (this.disposed) { return; }
        this.disposed = true;
        for (const buffer of this.buffers) { buffer.destroy(); }
        this.buffers.length = 0;
    }
}
