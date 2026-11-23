import { loadOfficialLayer } from "../../src/official-weights.js";
import { GEMMA3_1B } from "../../src/gemma3-core.js";
import { compareWindowCaches } from "../../src/gemma3-window-parity.js";

const directory = process.argv[2] ?? "fixtures/official-layer0";
const windowSize = Number(process.argv[3] ?? 8);
const targetLength = process.argv[4] === undefined ? null : Number(process.argv[4]);

if (!Number.isInteger(windowSize) || windowSize < 1
    || (targetLength !== null && (!Number.isInteger(targetLength) || targetLength < 1))) {
    throw new Error("Usage: node tests/official-window-boundary.mjs DIRECTORY [windowSize] [targetLength]");
}

const data = await loadOfficialLayer(directory);
const layerIndex = data.manifest.layerIndex;
const config = { ...GEMMA3_1B, slidingWindow: windowSize };

if (config.layerTypes[layerIndex] !== "local") {
    throw new Error("Please export a local-attention layer (e.g. layer 0)");
}

const sequenceLength = targetLength ?? data.manifest.sequenceLength;

if (sequenceLength <= windowSize) {
    throw new Error(`Need more than ${windowSize} input tokens to cross the window boundary`);
}

// Optional long stress test cycles real layer-input vectors. This tests cache
// mechanics with real weights, NOT parity against a natural long PyTorch prompt.
const inputs = Array.from({ length: sequenceLength }, (_, position) => {
    const sourcePosition = position % data.manifest.sequenceLength;
    return data.inputs.subarray(
        sourcePosition * config.hiddenSize,
        (sourcePosition + 1) * config.hiddenSize
    );
});

const result = compareWindowCaches(data.layer, inputs, config, layerIndex);
console.log(JSON.stringify({
    layerIndex,
    windowSize,
    sequenceLength: inputs.length,
    inputMode: targetLength === null ? "official-export" : "repeated-official-layer-inputs",
    configuration: windowSize === GEMMA3_1B.slidingWindow ? "official-window" : "shortened-window-test",
    ...result
}, null, 4));

if (!result.matches) {
    process.exitCode = 1;
}
