/** Token-parallel float32 prefill. The decode path remains unchanged.
 * Each layer computes a whole chunk, with a chronological K/V snapshot for
 * attention, then commits its new K/V to the decode ring after attention.
 */
const shader = `
struct Params {
    n: u32, m: u32, tokens: u32, base: u32,
    theta: f32, scalar: f32, eps: f32, history: u32,
    window: u32, headDim: u32, capacity: u32, reserved: u32,
};
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
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.tokens * p.m) { return; }
    let t = i / p.m;
    let row = i % p.m;
    var sum = 0.0;
    for (var j = 0u; j < p.n; j = j + 1u) {
        sum += w[row * p.n + j] * x[t * p.n + j];
    }
    y[i] = sum;
}
@compute @workgroup_size(64)
fn norm(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = group.x * p.n;
    var sum = 0.0;
    for (var j = lane; j < p.n; j += 64u) { sum += x[base + j] * x[base + j]; }
    let inv = inverseSqrt(sumGroup(sum, lane) / f32(p.n) + p.eps);
    for (var j = lane; j < p.n; j += 64u) {
        y[base + j] = x[base + j] * inv * (1.0 + w[j]);
    }
}
@compute @workgroup_size(64)
fn rope(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let dim = p.headDim;
    let base = group.x * dim;
    let half = dim / 2u;
    var sum = 0.0;
    for (var j = lane; j < dim; j += 64u) { sum += x[base + j] * x[base + j]; }
    let inv = inverseSqrt(sumGroup(sum, lane) / f32(dim) + p.eps);
    for (var j = lane; j < half; j += 64u) {
        let a = x[base + j] * inv * (1.0 + w[j]);
        let b = x[base + j + half] * inv * (1.0 + w[j + half]);
        let angle = f32(p.base + group.x / (p.n / dim)) / pow(p.theta, 2.0 * f32(j) / f32(dim));
        let cs = cos(angle);
        let sn = sin(angle);
        y[base + j] = a * cs - b * sn;
        y[base + j + half] = b * cs + a * sn;
    }
}
var<workgroup> scores: array<f32, 64>;
@compute @workgroup_size(64)
fn attention(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let dim = p.headDim;
    let heads = p.n / dim;
    let t = group.x / heads;
    let absolute = p.base + t;
    let firstAbsolute = p.base - p.history;
    let first = max(firstAbsolute, absolute + 1u - min(absolute + 1u, p.window)) - firstAbsolute;
    let count = p.history + t + 1u - first;
    let base = group.x * dim;
    var maximum = -3.402823e38;
    var denominator = 0.0;
    var accum = vec4<f32>(0.0);
    for (var tile = 0u; tile < count; tile += 64u) {
        let valid = min(64u, count - tile);
        var score = -3.402823e38;
        if (lane < valid) {
            let slot = first + tile + lane;
            var dot = 0.0;
            for (var j = 0u; j < dim; j += 1u) { dot += x[base + j] * w[slot * dim * 2u + j]; }
            score = dot / sqrt(p.scalar);
        }
        scores[lane] = score;
        workgroupBarrier();
        var nextMax = maximum;
        for (var k = 0u; k < valid; k += 1u) { nextMax = max(nextMax, scores[k]); }
        let factor = exp(maximum - nextMax);
        accum *= factor;
        denominator *= factor;
        workgroupBarrier();
        scores[lane] = exp(score - nextMax);
        workgroupBarrier();
        for (var k = 0u; k < valid; k += 1u) {
            let weight = scores[k];
            let offset = (first + tile + k) * dim * 2u + dim + lane * 4u;
            accum += weight * vec4<f32>(w[offset], w[offset + 1u], w[offset + 2u], w[offset + 3u]);
            denominator += weight;
        }
        maximum = nextMax;
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j += 1u) { y[base + lane * 4u + j] = accum[j] / denominator; }
}
@compute @workgroup_size(64)
fn copyvec(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < p.tokens * p.n) { y[id.x] = x[id.x]; }
}
@compute @workgroup_size(64)
fn addnorm(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = group.x * p.n;
    var sum = 0.0;
    for (var j = lane; j < p.n; j += 64u) { sum += x[base + j] * x[base + j]; }
    let inv = inverseSqrt(sumGroup(sum, lane) / f32(p.n) + p.eps);
    for (var j = lane; j < p.n; j += 64u) {
        y[base + j] = y[base + j] + x[base + j] * inv * (1.0 + w[j]);
    }
}
@compute @workgroup_size(64)
fn gelumul(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.tokens * p.n) { return; }
    let v = x[i];
    let a = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    y[i] = 0.5 * v * (1.0 + tanh(clamp(a, -10.0, 10.0))) * w[i];
}
`;

const halfMatrixShader = `enable f16;
struct Params {
    n: u32, m: u32, tokens: u32, base: u32,
    theta: f32, scalar: f32, eps: f32, history: u32,
    window: u32, headDim: u32, capacity: u32, reserved: u32,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> halfWeights: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64)
fn matmulHalf(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.tokens * p.m) { return; }
    let t = i / p.m;
    let row = i % p.m;
    var sum = 0.0;
    for (var j = 0u; j < p.n; j = j + 1u) {
        sum += f32(halfWeights[row * p.n + j]) * x[t * p.n + j];
    }
    y[i] = sum;
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
        let offset = (row * p.n / 32u + g) * 5u;
        var subtotal = 0.0;
        for (var word = 0u; word < 4u; word += 1u) {
            let packed = halfWeights[offset + 1u + word];
            for (var lane = 0u; lane < 8u; lane += 1u) {
                let q = i32((packed >> (lane * 4u)) & 15u) - 8;
                subtotal += f32(q) * x[t * p.n + g * 32u + word * 8u + lane];
            }
        }
        sum += subtotal * bitcast<f32>(halfWeights[offset]);
    }`);



function tiledMatrix(code, entry, weightExpression) {
    const start = code.indexOf('@compute @workgroup_size(64)\nfn ' + entry + '(');
    const body = code.indexOf('{', start);
    let end = body + 1;
    let depth = 1;
    while (depth > 0) {
        if (code[end] === '{') { depth += 1; }
        if (code[end] === '}') { depth -= 1; }
        end += 1;
    }
    const kernel = `
var<workgroup> tileX: array<f32, 64>;
var<workgroup> tileW: array<f32, 64>;
@compute @workgroup_size(8, 8)
fn ${entry}(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>) {
    let row = group.x * 8u + lane.x;
    let token = group.y * 8u + lane.y;
    var sum = 0.0;
    for (var base = 0u; base < p.n; base += 8u) {
        let kx = base + lane.x;
        let kw = base + lane.y;
        tileX[lane.y * 8u + lane.x] = 0.0;
        tileW[lane.y * 8u + lane.x] = 0.0;
        if (token < p.tokens && kx < p.n) {
            tileX[lane.y * 8u + lane.x] = x[token * p.n + kx];
        }
        if (row < p.m && kw < p.n) {
            tileW[lane.y * 8u + lane.x] = ${weightExpression};
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k += 1u) {
            sum += tileX[lane.y * 8u + k] * tileW[k * 8u + lane.x];
        }
        workgroupBarrier();
    }
    if (row < p.m && token < p.tokens) { y[token * p.m + row] = sum; }
}
`;
    return code.slice(0, start) + kernel + code.slice(end);
}

export class WebGpuParallelPrefill {
    constructor(device, config, halfWeights = false, int4Weights = false) {
        this.device = device;
        this.config = config;
        const module = device.createShaderModule({ code: tiledMatrix(shader, "matmul", "w[row * p.n + kw]") });
        this.pipelines = {};
        for (const name of ['matmul', 'norm', 'rope', 'attention', 'copyvec', 'addnorm', 'gelumul']) {
            this.pipelines[name] = device.createComputePipeline({
                layout: 'auto', compute: { module, entryPoint: name }
            });
        }
        if (halfWeights || int4Weights) {
            const halfModule = device.createShaderModule({ code: int4Weights ? tiledMatrix(int4MatrixShader, "matmulHalf", "weightAt(row * p.n + kw)") : tiledMatrix(halfMatrixShader, "matmulHalf", "f32(halfWeights[row * p.n + kw])") });
            this.pipelines.matmul = device.createComputePipeline({ layout: 'auto', compute: { module: halfModule, entryPoint: 'matmulHalf' } });
        }
    }

    /** One layer, multiple query positions, one chronological KV snapshot. */
    encodeLayer(encoder, layer, input, base, count, temporary) {
        const device = this.device;
        const c = this.config;
        const h = c.hiddenSize;
        const d = c.headDim;
        const intermediate = c.intermediateSize;
        const local = c.layerTypes[layer.layerIndex] === 'local';
        const history = local ? Math.min(base, c.slidingWindow - 1) : base;
        const keyCount = history + count;
        const stride = d * 8;
        const theta = local ? c.localRopeTheta : c.globalRopeTheta;
        const pool = this.pool ??= new Map();
        let slotIndex = 0;
        const make = length => {
            const bytes = length * 4;
            if (bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize) {
                throw new Error(`Parallel prefill needs ${bytes} bytes of storage; device limit exceeded`);
            }
            const slot = slotIndex++;
            const key = `${slot}:${bytes}:${slot === 9 ? layer.layerIndex % 2 : 0}`;
            const existing = pool.get(key);
            if (existing && temporary.includes(existing)) { return existing; }
            const buffer = device.createBuffer({
                size: Math.max(4, bytes),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            temporary.push(buffer);
            pool.set(key, buffer);
            return buffer;
        };
        const normed = make(count * h);
        const q = make(count * c.numHeads * d);
        const k = make(count * d);
        const v = make(count * d);
        const qr = make(count * c.numHeads * d);
        const kr = make(count * d);
        const kv = make(keyCount * d * 2);
        const att = make(count * c.numHeads * d);
        const projected = make(count * h);
        const ff = make(count * h);
        const gate = make(count * intermediate);
        const up = make(count * intermediate);
        const activated = make(count * intermediate);
        const down = make(count * h);
        const run = (entry, x, w, y, n, m, total = count, window = 0xffffffff) => {
            const params = new ArrayBuffer(48);
            const view = new DataView(params);
            for (const [offset, value] of [[0, n], [4, m], [8, total], [12, base], [28, history], [32, window], [36, d], [40, layer.maxCache]]) {
                view.setUint32(offset, value, true);
            }
            view.setFloat32(16, theta, true);
            view.setFloat32(20, c.queryPreAttnScalar, true);
            view.setFloat32(24, c.rmsNormEps, true);
            // Reuse the layer's aligned uniform arena. Each dispatch receives
            // its own record, so queued writes cannot overwrite parameters
            // used by another dispatch in this submission.
            if (layer.uniformCursor >= layer.uniformCapacity) {
                throw new Error(`Layer ${layer.layerIndex}: parallel prefill uniform arena exhausted`);
            }
            const uniformOffset = layer.uniformCursor * layer.uniformStride;
            layer.uniformCursor += 1;
            const uniform = layer.uniformArena;
            device.queue.writeBuffer(uniform, uniformOffset, params);
            const pipeline = this.pipelines[entry];
            const entries = (entry === 'copyvec'
                ? [[0, x], [2, y], [3, uniform]]
                : [[0, x], [1, w], [2, y], [3, uniform]]
            ).map(([binding, buffer]) => ({
                binding,
                resource: binding === 3
                    ? { buffer, offset: uniformOffset, size: 48 }
                    : { buffer }
            }));
            const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            const items = entry === 'matmul' ? m * total : n * total;
            if (entry === 'matmul') {
                pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(total / 8));
            } else if (entry === 'norm' || entry === 'addnorm') {
                pass.dispatchWorkgroups(total);
            } else if (entry === 'rope' || entry === 'attention') {
                pass.dispatchWorkgroups(total * n / d);
            } else {
                pass.dispatchWorkgroups(Math.ceil(items / 64));
            }
            pass.end();
        };
        run('norm', input, layer.weights.inputLayernorm, normed, h, h);
        run('matmul', normed, layer.weights.qProj, q, h, c.numHeads * d);
        run('matmul', normed, layer.weights.kProj, k, h, d);
        run('matmul', normed, layer.weights.vProj, v, h, d);
        run('rope', q, layer.weights.qNorm, qr, c.numHeads * d, d);
        run('rope', k, layer.weights.kNorm, kr, d, d);
        // Snapshot the old K/V before any ring slots are overwritten.
        const first = base - history;
        // A chronological snapshot takes at most two GPU copies, even when
        // the local ring wraps around. Global cache needs only one copy.
        if (history > 0) {
            const firstSlot = first % layer.maxCache;
            const firstRun = Math.min(history, layer.maxCache - firstSlot);
            encoder.copyBufferToBuffer(layer.kv, firstSlot * stride, kv, 0, firstRun * stride);
            if (firstRun < history) {
                encoder.copyBufferToBuffer(layer.kv, 0, kv, firstRun * stride, (history - firstRun) * stride);
            }
        }
        for (let t = 0; t < count; t += 1) {
            encoder.copyBufferToBuffer(kr, t * d * 4, kv, (history + t) * stride, d * 4);
            encoder.copyBufferToBuffer(v, t * d * 4, kv, (history + t) * stride + d * 4, d * 4);
        }
        run('attention', qr, kv, att, c.numHeads * d, d, count, local ? c.slidingWindow : 0xffffffff);
        // Commit the last cache-capacity tokens in order, after attention.
        for (let t = 0; t < count; t += 1) {
            const slot = (base + t) % layer.maxCache;
            encoder.copyBufferToBuffer(kr, t * d * 4, layer.kv, slot * stride, d * 4);
            encoder.copyBufferToBuffer(v, t * d * 4, layer.kv, slot * stride + d * 4, d * 4);
        }
        run('matmul', att, layer.weights.oProj, projected, c.numHeads * d, h);
        run('copyvec', input, layer.weights.inputLayernorm, ff, h, h);
        run('addnorm', projected, layer.weights.postAttentionLayernorm, ff, h, h);
        run('norm', ff, layer.weights.preFeedforwardLayernorm, normed, h, h);
        run('matmul', normed, layer.weights.gateProj, gate, h, intermediate);
        run('matmul', normed, layer.weights.upProj, up, h, intermediate);
        run('gelumul', gate, up, activated, intermediate, intermediate);
        run('matmul', activated, layer.weights.downProj, down, intermediate, h);
        run('addnorm', down, layer.weights.postFeedforwardLayernorm, ff, h, h);
        return ff;
    }
}
