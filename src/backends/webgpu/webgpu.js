/** Minimal WGSL matrix-vector projection, weights in row-major [out, in].
 * Standalone experimental kernel. Not yet wired into the full decoder.
 */
const MATVEC_SHADER = `
struct Params { inputSize: u32, outputSize: u32, };
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.outputSize) { return; }
    var total = 0.0;
    for (var column: u32 = 0u; column < params.inputSize; column = column + 1u) {
        total = total + weights[row * params.inputSize + column] * input[column];
    }
    output[row] = total;
}
`;

export class WebGpuBackend {
    constructor(device) {
        this.device = device;
        this.pipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module: device.createShaderModule({ code: MATVEC_SHADER }), entryPoint: "main" }
        });
    }

    static async create() {
        if (!navigator.gpu) {
            throw new Error("WebGPU unavailable");
        }

        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter?.requestDevice();

        if (!device) {
            throw new Error("WebGPU device unavailable");
        }

        return new WebGpuBackend(device);
    }

    async linear(vector, weights, outputSize, inputSize) {
        if (vector.length !== inputSize || weights.length !== outputSize * inputSize) {
            throw new Error("WebGpuBackend.linear: dimension mismatch");
        }

        const device = this.device;
        const buffers = [];
        const allocate = (size, usage) => {
            const buffer = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
            buffers.push(buffer);
            return buffer;
        };

        try {
            const w = allocate(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const x = allocate(vector.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const y = allocate(outputSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const p = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const readback = allocate(outputSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            device.queue.writeBuffer(w, 0, weights);
            device.queue.writeBuffer(x, 0, Float32Array.from(vector));
            device.queue.writeBuffer(p, 0, new Uint32Array([inputSize, outputSize, 0, 0]));

            const bindGroup = device.createBindGroup({
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [w, x, y, p].map((buffer, binding) => ({ binding, resource: { buffer } }))
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(outputSize / 64));
            pass.end();
            encoder.copyBufferToBuffer(y, 0, readback, 0, outputSize * 4);
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
}
