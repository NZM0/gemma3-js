import { test } from "node:test";
import { existsSync } from "node:fs";
import { GEMMA3_1B } from "../../src/gemma3-core.js";
import { Gemma3Model } from "../../src/gemma3-model.js";
import { loadOfficialModel } from "../../src/official-model-weights.js";

function argmax(values) {
    let best = 0;
    for (let i = 1; i < values.length; i += 1) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

const directory = process.argv[2] ?? "fixtures/official-model";
if (!existsSync(`${directory}/manifest.json`)) {
    test.skip("Official model comparison: reference fixture not installed");
} else {

const { manifest, layers, embeddings, finalNorm, referenceHidden, headRows, referenceLogits } =
    await loadOfficialModel(directory);
const config = { ...GEMMA3_1B, numLayers: layers.length };
const model = new Gemma3Model(layers, finalNorm, config);
const hidden = config.hiddenSize;
const tokens = manifest.sequenceLength;
const stats = Array.from({ length: layers.length }, () => ({ max: 0, sumSq: 0, count: 0 }));
const result = model.forwardEmbeddings(embeddings, (layer, position, values) => {
    const start = ((layer + 1) * tokens + position) * hidden;
    const stat = stats[layer];
    for (let i = 0; i < hidden; i += 1) {
        const error = Math.abs(values[i] - referenceHidden[start + i]);
        if (!Number.isFinite(error)) throw new Error(`Nonfinite layer ${layer} position ${position}`);
        stat.max = Math.max(stat.max, error);
        stat.sumSq += error * error;
        stat.count += 1;
    }
});

const logits = model.logitsForRows(result.lastNormalized, headRows);
let maxLogitError = 0;
for (let i = 0; i < logits.length; i += 1) {
    maxLogitError = Math.max(maxLogitError, Math.abs(logits[i] - referenceLogits[i]));
}
console.log(JSON.stringify({
    tokens,
    layers: stats.map((stat, index) => ({
        index,
        maxAbsoluteError: stat.max,
        rmsError: Math.sqrt(stat.sumSq / stat.count),
    })),
    headMode: manifest.head.mode,
    maxLogitError,
    predictedTokenIdWithinExportedRows: manifest.head.mode === "full"
        ? argmax(logits)
        : manifest.head.tokenIds[argmax(logits)],
    referenceTop10: manifest.referenceTop10,
}, null, 4));

const toleranceIndex = process.argv.indexOf("--tolerance");
if (toleranceIndex >= 0) {
    const tolerance = Number(process.argv[toleranceIndex + 1]);
    if (!Number.isFinite(tolerance) || tolerance < 0) throw new Error("Invalid tolerance");
    const maxLayerError = Math.max(...stats.map(stat => stat.max));
    if (maxLayerError > tolerance) throw new Error(`Layer max error ${maxLayerError} > ${tolerance}`);
    console.log("Full-model layer tolerance: PASS");
}

}
