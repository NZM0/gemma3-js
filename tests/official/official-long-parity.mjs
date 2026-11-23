import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { GEMMA3_1B, SessionCache, rmsNorm } from "../../src/gemma3-core.js";
import { decoderStep } from "../../src/gemma3-decoder.js";
import { loadOfficialModel, readF32 } from "../../src/official-model-weights.js";

const weightsDir = process.argv[2] ?? "fixtures/official-model-full";
const referenceDir = process.argv[3] ?? "fixtures/long-reference";
const limitIndex = process.argv.indexOf("--max-positions");
const maxPositions = limitIndex < 0 ? Infinity : Number(process.argv[limitIndex + 1]);

if (!existsSync(join(weightsDir, "manifest.json")) || !existsSync(join(referenceDir, "manifest.json"))) {
    throw new Error("Missing official weights or long reference; see README.md");
}
if (!(maxPositions > 0)) throw new Error("--max-positions must be positive");

const oracle = JSON.parse(readFileSync(join(referenceDir, "manifest.json"), "utf8"));
const { manifest, layers, finalNorm, headRows } = await loadOfficialModel(weightsDir);
if (manifest.head.mode !== "full") throw new Error("Full-vocab head is required");
if (oracle.numLayers !== layers.length || oracle.hiddenSize !== GEMMA3_1B.hiddenSize || oracle.windowSize !== 512) {
    throw new Error("Incompatible oracle / model configuration");
}

const hiddenSize = oracle.hiddenSize;
const embeddings = await readF32(join(referenceDir, oracle.embeddingFile), oracle.length * hiddenSize);
const expected = await readF32(
    join(referenceDir, oracle.hiddenFile),
    (layers.length + 1) * oracle.checkpoints.length * hiddenSize
);
const expectedLogits = await readF32(join(referenceDir, oracle.logitsFile), oracle.vocabSize);
const positions = new Map(oracle.checkpoints.map((position, index) => [position, index]));
const stats = oracle.checkpoints.map(position => ({
    position,
    layers: Array.from({ length: layers.length + 1 }, () => ({ max: 0, sumSq: 0, count: 0 }))
}));
const cache = new SessionCache(layers.length);
let lastHidden = null;
const count = Math.min(oracle.length, maxPositions);

function compare(layerIndex, position, values) {
    const checkpointIndex = positions.get(position);
    if (checkpointIndex === undefined) return;
    const stat = stats[checkpointIndex].layers[layerIndex];
    const start = (layerIndex * positions.size + checkpointIndex) * hiddenSize;
    for (let i = 0; i < hiddenSize; i += 1) {
        const error = Math.abs(values[i] - expected[start + i]);
        if (!Number.isFinite(error)) throw new Error(`Nonfinite error at layer ${layerIndex}, position ${position}`);
        stat.max = Math.max(stat.max, error);
        stat.sumSq += error * error;
        stat.count += 1;
    }
}

console.log(`Official long parity: ${count}/${oracle.length} positions, ${layers.length} layers, 512 local window`);
for (let position = 0; position < count; position += 1) {
    let hidden = embeddings.subarray(position * hiddenSize, (position + 1) * hiddenSize);
    compare(0, position, hidden);
    for (let layer = 0; layer < layers.length; layer += 1) {
        hidden = decoderStep(hidden, layers[layer], cache, layer, position, GEMMA3_1B);
        compare(layer + 1, position, hidden);
    }
    lastHidden = hidden;
    if (position % 25 === 0 || position === count - 1) {
        console.log(`Processed ${position + 1}/${count} positions`);
    }
}

const summaries = stats.filter(item => item.layers.some(stat => stat.count)).map(item => ({
    position: item.position,
    maxAbsoluteError: Math.max(...item.layers.map(stat => stat.max)),
    lastLayerRmsError: Math.sqrt(
        item.layers.at(-1).sumSq / (item.layers.at(-1).count || 1)
    ),
    worstLayer: item.layers.reduce((best, stat, index, array) =>
        stat.max > array[best].max ? index : best, 0),
}));

let logitsResult = null;
if (count === oracle.length) {
    const normalized = rmsNorm(lastHidden, finalNorm, GEMMA3_1B.rmsNormEps);
    const scores = new Float32Array(oracle.vocabSize);
    let maxError = 0;
    let predictedId = 0;
    for (let token = 0; token < oracle.vocabSize; token += 1) {
        let sum = 0;
        const offset = token * hiddenSize;
        for (let i = 0; i < hiddenSize; i += 1) sum += headRows[offset + i] * normalized[i];
        scores[token] = sum;
        maxError = Math.max(maxError, Math.abs(scores[token] - expectedLogits[token]));
        if (scores[token] > scores[predictedId]) predictedId = token;
    }
    logitsResult = {
        maxAbsoluteError: maxError,
        predictedTokenId: predictedId,
        referenceTop1: oracle.referenceTop10[0],
        top1Matches: predictedId === oracle.referenceTop10[0],
    };
}

console.log(JSON.stringify({
    positionsProcessed: count,
    checkpoints: summaries,
    logits: logitsResult,
    complete: count === oracle.length,
}, null, 4));

const toleranceIndex = process.argv.indexOf("--tolerance");
if (toleranceIndex >= 0) {
    const tolerance = Number(process.argv[toleranceIndex + 1]);
    if (!Number.isFinite(tolerance) || tolerance < 0) throw new Error("Invalid tolerance");
    const maxError = Math.max(...summaries.map(item => item.maxAbsoluteError));
    if (maxError > tolerance) throw new Error(`Checkpoint max error ${maxError} > ${tolerance}`);
    if (logitsResult && !logitsResult.top1Matches) throw new Error("Top-1 mismatch");
    console.log("Long parity tolerance: PASS");
}
