import { rmsNorm, geluTanh, applyRopeHalfSplit, attentionForHead, SessionCache, GEMMA3_1B } from "../../gemma3-core.js";

const names = ["qProj", "kProj", "vProj", "oProj", "gateProj", "upProj", "downProj"];

const shader = `
struct Params { columns: u32, rows: u32, };
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = 0u; col < params.columns; col = col + 1u) {
        sum += weights[row * params.columns + col] * input[col];
    }
    output[row] = sum;
}
`;

function splitHeads(vector, count, size) {
    return Array.from({ length: count }, (_, i) => vector.slice(i * size, (i + 1) * size));
}

function add(a, b) {
    return Float32Array.from(a, (value, i) => value + b[i]);
}

/** Stage 12: persistent WGSL matvec weights, CPU norm/RoPE/attention/activation. */
export class WebGpuLayerSession {
    constructor(device, layer, config = GEMMA3_1B, layerIndex = 0) {
        this.device = device;
        this.layer = layer;
        this.config = config;
        this.layerIndex = layerIndex;
        this.position = 0;
        this.cache = new SessionCache(config.numLayers);
        this.resources = new Map();
        this.disposed = false;

        const shapes = {
            qProj: [config.numHeads * config.headDim, config.hiddenSize],
            kProj: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
            vProj: [config.numKeyValueHeads * config.headDim, config.hiddenSize],
            oProj: [config.hiddenSize, config.numHeads * config.headDim],
            gateProj: [config.intermediateSize, config.hiddenSize],
            upProj: [config.intermediateSize, config.hiddenSize],
            downProj: [config.hiddenSize, config.intermediateSize]
        };

        try {
            const module = device.createShaderModule({ code: shader });
            this.pipeline = device.createComputePipeline({
                layout: "auto",
                compute: { module, entryPoint: "main" }
            });

            for (const name of names) {
                const [rows, columns] = shapes[name];
                const data = layer[name];
                if (data.length !== rows * columns) {
                    throw new Error(`${name}: incompatible weight shape`);
                }
                const bytes = data.byteLength;
                if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize) {
                    throw new Error(`${name}: ${bytes} bytes exceeds device storage-buffer limit`);
                }
                const weight = device.createBuffer({
                    size: bytes,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                device.queue.writeBuffer(weight, 0, data);
                this.resources.set(name, { weight, rows, columns });
            }
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    async project(name, input) {
        if (this.disposed) {
            throw new Error("WebGpuLayerSession has been disposed");
        }
        const device = this.device;
        const { weight, rows, columns } = this.resources.get(name);
        if (input.length !== columns) {
            throw new Error(`${name}: expected input length ${columns}, got ${input.length}`);
        }
        const buffers = [];
        const allocate = (size, usage) => {
            const buffer = device.createBuffer({ size: Math.max(4, size), usage });
            buffers.push(buffer);
            return buffer;
        };

        try {
            const x = allocate(columns * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const y = allocate(rows * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const params = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const readback = allocate(rows * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            device.queue.writeBuffer(x, 0, input);
            device.queue.writeBuffer(params, 0, new Uint32Array([columns, rows, 0, 0]));

            const group = device.createBindGroup({
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [weight, x, y, params].map((buffer, binding) => ({
                    binding,
                    resource: { buffer }
                }))
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil(rows / 64));
            pass.end();
            encoder.copyBufferToBuffer(y, 0, readback, 0, rows * 4);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(readback.getMappedRange().slice(0));
            readback.unmap();
            return result;
        } finally {
            for (const buffer of buffers) {
                buffer.destroy();
            }
        }
    }

    async step(input) {
        if (this.disposed) {
            throw new Error("WebGpuLayerSession has been disposed");
        }
        const config = this.config;
        const layer = this.layer;
        const position = this.position;
        const index = this.layerIndex;
        const layerType = config.layerTypes[index];
        const theta = layerType === "local" ? config.localRopeTheta : config.globalRopeTheta;
        const normalized = rmsNorm(input, layer.inputLayernorm, config.rmsNormEps);

        const [qFlat, kFlat, vFlat] = await Promise.all([
            this.project("qProj", normalized),
            this.project("kProj", normalized),
            this.project("vProj", normalized)
        ]);
        const q = splitHeads(qFlat, config.numHeads, config.headDim);
        const k = splitHeads(kFlat, config.numKeyValueHeads, config.headDim);
        const v = splitHeads(vFlat, config.numKeyValueHeads, config.headDim);
        const queries = q.map(head => applyRopeHalfSplit(rmsNorm(head, layer.qNorm, config.rmsNormEps), position, theta));
        const keys = k.map(head => applyRopeHalfSplit(rmsNorm(head, layer.kNorm, config.rmsNormEps), position, theta));

        this.cache.append(index, position, keys, v, layerType, config.slidingWindow);
        const state = this.cache.layers[index];
        const heads = queries.map((query, qIndex) => {
            const kvIndex = Math.floor(qIndex / (config.numHeads / config.numKeyValueHeads));
            return attentionForHead(
                query,
                state.keys.map(entry => entry[kvIndex]),
                state.values.map(entry => entry[kvIndex]),
                state.positions,
                position,
                layerType,
                config.slidingWindow,
                config.queryPreAttnScalar
            );
        });
        const combined = Float32Array.from(heads.flatMap(head => Array.from(head)));
        const attention = await this.project("oProj", combined);
        const afterAttention = add(input, rmsNorm(attention, layer.postAttentionLayernorm, config.rmsNormEps));
        const ffInput = rmsNorm(afterAttention, layer.preFeedforwardLayernorm, config.rmsNormEps);
        const [gate, up] = await Promise.all([
            this.project("gateProj", ffInput),
            this.project("upProj", ffInput)
        ]);
        const activated = Float32Array.from(gate, (value, i) => geluTanh(value) * up[i]);
        const down = await this.project("downProj", activated);
        const output = add(afterAttention, rmsNorm(down, layer.postFeedforwardLayernorm, config.rmsNormEps));
        this.position += 1;
        return output;
    }

    reset() {
        this.cache.clear();
        this.position = 0;
    }

    dispose() {
        if (this.disposed) {
            return;
        }
        for (const resource of this.resources.values()) {
            resource.weight.destroy();
        }
        this.resources.clear();
        this.disposed = true;
    }
}
