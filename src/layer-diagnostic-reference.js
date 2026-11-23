import { rmsNorm, geluTanh, applyRopeHalfSplit, attentionForHead, SessionCache } from "./gemma3-core.js";
import { linear } from "./gemma3-decoder.js";

export class CpuLayerDiagnostic {
    constructor(layer, config, layerIndex = 0) {
        this.layer = layer;
        this.config = config;
        this.layerIndex = layerIndex;
        this.position = 0;
        this.cache = new SessionCache(config.numLayers);
    }

    step(input) {
        const c = this.config;
        const l = this.layer;
        const d = c.headDim;
        const position = this.position;
        const type = c.layerTypes[this.layerIndex];
        const theta = type === "local" ? c.localRopeTheta : c.globalRopeTheta;
        const trace = {};

        trace.inputNorm = rmsNorm(input, l.inputLayernorm, c.rmsNormEps);
        trace.qProjection = linear(trace.inputNorm, l.qProj, c.numHeads * d, c.hiddenSize);
        trace.kProjection = linear(trace.inputNorm, l.kProj, c.numKeyValueHeads * d, c.hiddenSize);
        trace.vProjection = linear(trace.inputNorm, l.vProj, c.numKeyValueHeads * d, c.hiddenSize);

        const qHeads = Array.from({ length: c.numHeads }, (_, i) => trace.qProjection.slice(i * d, (i + 1) * d));
        const kHeads = Array.from({ length: c.numKeyValueHeads }, (_, i) => trace.kProjection.slice(i * d, (i + 1) * d));
        const vHeads = Array.from({ length: c.numKeyValueHeads }, (_, i) => trace.vProjection.slice(i * d, (i + 1) * d));
        const qRopeHeads = qHeads.map(head => applyRopeHalfSplit(rmsNorm(head, l.qNorm, c.rmsNormEps), position, theta));
        const kRopeHeads = kHeads.map(head => applyRopeHalfSplit(rmsNorm(head, l.kNorm, c.rmsNormEps), position, theta));
        trace.qRope = Float32Array.from(qRopeHeads.flatMap(head => Array.from(head)));
        trace.kRope = Float32Array.from(kRopeHeads.flatMap(head => Array.from(head)));

        this.cache.append(this.layerIndex, position, kRopeHeads, vHeads, type, c.slidingWindow);
        const state = this.cache.layers[this.layerIndex];
        const attentionHeads = qRopeHeads.map((query, qIndex) => {
            const kvIndex = Math.floor(qIndex / (c.numHeads / c.numKeyValueHeads));
            return attentionForHead(
                query,
                state.keys.map(keys => keys[kvIndex]),
                state.values.map(values => values[kvIndex]),
                state.positions,
                position,
                type,
                c.slidingWindow,
                c.queryPreAttnScalar
            );
        });
        trace.attention = Float32Array.from(attentionHeads.flatMap(head => Array.from(head)));
        trace.oProjection = linear(trace.attention, l.oProj, c.hiddenSize, c.numHeads * d);
        const attentionNorm = rmsNorm(trace.oProjection, l.postAttentionLayernorm, c.rmsNormEps);
        trace.afterAttention = Float32Array.from(input, (value, i) => value + attentionNorm[i]);
        trace.ffNorm = rmsNorm(trace.afterAttention, l.preFeedforwardLayernorm, c.rmsNormEps);
        trace.gate = linear(trace.ffNorm, l.gateProj, c.intermediateSize, c.hiddenSize);
        trace.up = linear(trace.ffNorm, l.upProj, c.intermediateSize, c.hiddenSize);
        trace.activated = Float32Array.from(trace.gate, (value, i) => geluTanh(value) * trace.up[i]);
        trace.down = linear(trace.activated, l.downProj, c.hiddenSize, c.intermediateSize);
        const finalNorm = rmsNorm(trace.down, l.postFeedforwardLayernorm, c.rmsNormEps);
        trace.output = Float32Array.from(trace.afterAttention, (value, i) => value + finalNorm[i]);

        this.position += 1;
        return trace;
    }
}

export function compareTraces(cpu, gpu) {
    const results = [];
    for (const [name, reference] of Object.entries(cpu)) {
        const actual = gpu[name];
        if (!actual || actual.length !== reference.length) {
            throw new Error(`Missing/mismatched diagnostic tensor: ${name}`);
        }
        let maxAbs = 0;
        let squared = 0;
        let worstIndex = -1;
        let nonFinite = 0;
        for (let i = 0; i < reference.length; i += 1) {
            if (!Number.isFinite(actual[i]) || !Number.isFinite(reference[i])) {
                nonFinite += 1;
                continue;
            }
            const error = Math.abs(actual[i] - reference[i]);
            squared += error * error;
            if (error > maxAbs) {
                maxAbs = error;
                worstIndex = i;
            }
        }
        results.push({ name, length: reference.length, maxAbs, rmsError: Math.sqrt(squared / reference.length), worstIndex, nonFinite });
    }
    return results;
}
