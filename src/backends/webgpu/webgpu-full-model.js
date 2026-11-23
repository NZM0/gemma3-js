import { GEMMA3_1B, rmsNorm } from "../../gemma3-core.js";
import { WebGpuFullLayerSession } from "./webgpu-full-layer.js";
import { WebGpuParallelPrefill } from "./webgpu-parallel-prefill.js";

/** Stage 14: all 26 decoder layers on GPU; one submission and one readback per token.
 * Final RMSNorm and optional selected-head logits remain on CPU for parity.
 */
export class WebGpuFullModelSession {
    constructor(device, layers, finalNorm, config = GEMMA3_1B) {
        if (layers.length !== config.numLayers) {
            throw new Error(`Expected ${config.numLayers} layers, got ${layers.length}`);
        }
        this.device = device;
        this.config = config;
        this.finalNorm = finalNorm;
        this.position = 0;
        this.disposed = false;
        this.layers = [];
        this.input = null;
        this.readback = null;
        this.parallelPrefill = null;
        this.gpuOutputEnabled = false;
        this.gpuOutput = null;
        try {
            for (let index = 0; index < layers.length; index += 1) {
                this.layers.push(layers[index] instanceof WebGpuFullLayerSession
                    ? layers[index]
                    : new WebGpuFullLayerSession(device, layers[index], config, index));
            }
            const bytes = config.hiddenSize * 4;
            this.input = device.createBuffer({
                size: bytes,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            this.readback = device.createBuffer({
                size: bytes,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    enableGpuOutput() {
        if (this.disposed) { throw new Error("Session disposed"); }
        if (!this.gpuOutput) {
            this.gpuOutput = this.device.createBuffer({
                size: this.config.hiddenSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
        }
        this.gpuOutputEnabled = true;
    }

    async step(input, options = {}) {
        if (this.disposed) { throw new Error("Session disposed"); }
        if (!(input instanceof Float32Array) || input.length !== this.config.hiddenSize) {
            throw new Error(`Expected Float32Array(${this.config.hiddenSize})`);
        }
        const trace = options.trace === true;
        const snapshots = [];
        const temporary = [];
        const encoder = this.device.createCommandEncoder();
        this.device.queue.writeBuffer(this.input, 0, input);
        let hidden = this.input;
        try {
            for (let index = 0; index < this.layers.length; index += 1) {
                const layer = this.layers[index];
                hidden = layer.encodeStep(encoder, hidden, this.position, temporary);
                if (trace) {
                    const staging = this.device.createBuffer({
                        size: this.config.hiddenSize * 4,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                    });
                    temporary.push(staging);
                    encoder.copyBufferToBuffer(hidden, 0, staging, 0, this.config.hiddenSize * 4);
                    snapshots.push(staging);
                }
            }
            const gpuOnly = this.gpuOutputEnabled && !trace;
            encoder.copyBufferToBuffer(hidden, 0, gpuOnly ? this.gpuOutput : this.readback, 0, this.config.hiddenSize * 4);
            this.device.queue.submit([encoder.finish()]);
            let output = null;
            if (gpuOnly) {
                await this.device.queue.onSubmittedWorkDone();
            } else {
                await this.readback.mapAsync(GPUMapMode.READ);
                output = new Float32Array(this.readback.getMappedRange().slice(0));
                this.readback.unmap();
            }
            const layerOutputs = [];
            for (const staging of snapshots) {
                await staging.mapAsync(GPUMapMode.READ);
                layerOutputs.push(new Float32Array(staging.getMappedRange().slice(0)));
                staging.unmap();
            }
            this.position += 1;
            for (const layer of this.layers) { layer.position += 1; }
            if (gpuOnly) { return { gpuHiddenBuffer: this.gpuOutput }; }
            const lastNormalized = rmsNorm(output, this.finalNorm, this.config.rmsNormEps);
            return trace ? { output, lastNormalized, layerOutputs } : { output, lastNormalized };
        } finally {
            // Readback completion ensures all commands in this submission have finished.
            for (const buffer of temporary) { buffer?.destroy(); }
            const dead = new Set(temporary);
            for (const layer of this.layers) {
                layer.buffers = layer.buffers.filter(buffer => !dead.has(buffer));
            }
        }
    }

    /**
     * Execute a prompt in GPU command batches without a CPU/GPU readback per token.
     * This is a scheduling optimization, not yet a token-parallel matmul kernel.
     * The existing causal/local-window attention and per-layer K/V are preserved.
     * A chunk's commands execute in order, so local ring slots are not overwritten
     * before the earlier queries have consumed them.
     */
    async prefill(embeddings, options = {}) {
        if (this.disposed) { throw new Error("Session disposed"); }
        if (!Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("prefill expects a nonempty array of embeddings");
        }
        if (options.trace === true) {
            let result = null;
            for (const embedding of embeddings) {
                result = await this.step(embedding, options);
            }
            return result;
        }
        if (options.parallel !== false && this.layers.every(layer => layer instanceof WebGpuFullLayerSession && layer.kv && layer.weights)) {
            return this.prefillParallel(embeddings, options);
        }
        const batchSize = options.batchSize ?? 16;
        if (!Number.isInteger(batchSize) || batchSize < 1) {
            throw new Error("batchSize must be a positive integer");
        }
        const h = this.config.hiddenSize;
        const bytes = h * 4;
        for (const embedding of embeddings) {
            if (!(embedding instanceof Float32Array) || embedding.length !== h) {
                throw new Error(`Expected Float32Array(${h})`);
            }
        }
        // Validate global capacity before encoding anything, so a failure does not
        // leave a partially updated K/V cache.
        const endPosition = this.position + embeddings.length;
        for (const layer of this.layers) {
            if (this.config.layerTypes[layer.layerIndex] === "global" && endPosition > layer.maxCache) {
                throw new Error(`Layer ${layer.layerIndex}: global KV cache capacity exceeded`);
            }
        }
        let result = null;
        for (let start = 0; start < embeddings.length; start += batchSize) {
            const count = Math.min(batchSize, embeddings.length - start);
            const temporary = [];
            const encoder = this.device.createCommandEncoder();
            for (const layer of this.layers) {
                layer.prepareBatch(count);
            }
            try {
                for (let index = 0; index < count; index += 1) {
                    const staging = this.device.createBuffer({
                        size: bytes,
                        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                    });
                    temporary.push(staging);
                    this.device.queue.writeBuffer(staging, 0, embeddings[start + index]);
                    encoder.copyBufferToBuffer(staging, 0, this.input, 0, bytes);
                    let hidden = this.input;
                    const position = this.position + index;
                    for (const layer of this.layers) {
                        // encodeStep checks the current logical position. The
                        // batch advances it only on the CPU until submission.
                        layer.position = position;
                        hidden = layer.encodeStep(encoder, hidden, position, temporary, true);
                    }
                    if (index === count - 1) {
                        encoder.copyBufferToBuffer(hidden, 0, this.gpuOutputEnabled ? this.gpuOutput : this.readback, 0, bytes);
                    }
                }
                this.device.queue.submit([encoder.finish()]);
                let output = null;
                if (this.gpuOutputEnabled) {
                    await this.device.queue.onSubmittedWorkDone();
                } else {
                    await this.readback.mapAsync(GPUMapMode.READ);
                    output = new Float32Array(this.readback.getMappedRange().slice(0));
                    this.readback.unmap();
                }
                this.position += count;
                for (const layer of this.layers) { layer.position = this.position; }
                result = this.gpuOutputEnabled
                    ? { gpuHiddenBuffer: this.gpuOutput }
                    : { output, lastNormalized: rmsNorm(output, this.finalNorm, this.config.rmsNormEps) };
            } finally {
                for (const buffer of temporary) { buffer?.destroy(); }
                const dead = new Set(temporary);
                for (const layer of this.layers) {
                    layer.buffers = layer.buffers.filter(buffer => !dead.has(buffer));
                }
            }
        }
        return result;
    }

    /** True token-parallel prefill: each transformer layer processes a chunk
     * as a matrix. Attention is causal and respects the official local window.
     * Existing per-layer K/V ring buffers are updated for decode continuation.
     */
    async prefillParallel(embeddings, options = {}) {
        if (this.disposed) { throw new Error("Session disposed"); }
        if (!Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("prefillParallel expects a nonempty array");
        }
        const countLimit = options.batchSize ?? 16;
        if (!Number.isInteger(countLimit) || countLimit < 1 || countLimit > this.config.slidingWindow) {
            throw new Error(`Parallel batchSize must be in 1..${this.config.slidingWindow}`);
        }
        const h = this.config.hiddenSize;
        for (const embedding of embeddings) {
            if (!(embedding instanceof Float32Array) || embedding.length !== h) {
                throw new Error(`Expected Float32Array(${h})`);
            }
        }
        const endPosition = this.position + embeddings.length;
        for (const layer of this.layers) {
            if (this.config.layerTypes[layer.layerIndex] === "global" && endPosition > layer.maxCache) {
                throw new Error(`Layer ${layer.layerIndex}: global KV cache capacity exceeded`);
            }
        }
        this.parallelPrefill ??= new WebGpuParallelPrefill(this.device, this.config);
        let result = null;
        for (let start = 0; start < embeddings.length; start += countLimit) {
            const count = Math.min(countLimit, embeddings.length - start);
            const temporary = [];
            const packed = new Float32Array(count * h);
            for (let i = 0; i < count; i += 1) {
                packed.set(embeddings[start + i], i * h);
            }
            const input = this.device.createBuffer({
                size: packed.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            temporary.push(input);
            this.device.queue.writeBuffer(input, 0, packed);
            const encoder = this.device.createCommandEncoder();
            let hidden = input;
            try {
                // Previous chunk has completed before these records are reused.
                for (const layer of this.layers) {
                    layer.prepareBatch(1);
                }
                for (const layer of this.layers) {
                    hidden = this.parallelPrefill.encodeLayer(
                        encoder, layer, hidden, this.position, count, temporary
                    );
                }
                encoder.copyBufferToBuffer(hidden, (count - 1) * h * 4, this.gpuOutputEnabled ? this.gpuOutput : this.readback, 0, h * 4);
                this.device.queue.submit([encoder.finish()]);
                let output = null;
                if (this.gpuOutputEnabled) {
                    await this.device.queue.onSubmittedWorkDone();
                } else {
                    await this.readback.mapAsync(GPUMapMode.READ);
                    output = new Float32Array(this.readback.getMappedRange().slice(0));
                    this.readback.unmap();
                }
                this.position += count;
                for (const layer of this.layers) { layer.position = this.position; }
                result = this.gpuOutputEnabled
                    ? { gpuHiddenBuffer: this.gpuOutput }
                    : { output, lastNormalized: rmsNorm(output, this.finalNorm, this.config.rmsNormEps) };
            } finally {
                for (const buffer of temporary) { buffer.destroy(); }
            }
        }
        return result;
    }

    /**
     * Reference full-prefix forward without reusing K/V from earlier calls.
     *
     * Replays all embeddings from position zero. The existing attention kernels
     * use temporary K/V scratch during this one forward; no K/V survives as a
     * reusable prefix across invocations. This is an independent execution
     * schedule, NOT an independent attention-kernel implementation.
     *
     * This O(sequenceLength^2) reference path is intentionally slow and should
     * only be used for short parity fixtures, not interactive generation.
     */
    async forwardNoCache(embeddings, options = {}) {
        if (!Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("forwardNoCache expects a nonempty array of embeddings");
        }
        this.reset();
        let result = null;
        try {
            for (const embedding of embeddings) {
                result = await this.step(embedding, options);
            }
            return result;
        } finally {
            // No reusable prefix state is retained after the forward.
            this.reset();
        }
    }

    reset() {
        this.position = 0;
        for (const layer of this.layers) { layer.reset(); }
    }

    dispose() {
        if (this.disposed) { return; }
        this.disposed = true;
        for (const layer of this.layers) { layer.dispose(); }
        this.input?.destroy();
        this.readback?.destroy();
        this.gpuOutput?.destroy();
    }
}
