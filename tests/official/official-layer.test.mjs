import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { DecoderSession } from "../../src/gemma3-decoder.js";
import { GEMMA3_1B } from "../../src/gemma3-core.js";
import { loadOfficialLayer } from "../../src/official-weights.js";

const directory = process.argv[2] ?? "fixtures/official-layer0";

if (!existsSync(`${directory}/manifest.json`)) {
    test.skip("Official layer comparison: reference fixture not installed");
} else {

const { manifest, layer, inputs, expected } = await loadOfficialLayer(directory);
const index = manifest.layerIndex;
const config = {
    ...GEMMA3_1B,
    numLayers: 1,
    layerTypes: [GEMMA3_1B.layerTypes[index]],
};
const session = new DecoderSession(config, [layer]);
const hidden = manifest.hiddenSize;
let maxError = 0;
let sumSquares = 0;
let count = 0;

for (let position = 0; position < manifest.sequenceLength; position += 1) {
    const start = position * hidden;
    const actual = session.step(inputs.subarray(start, start + hidden));

    for (let j = 0; j < hidden; j += 1) {
        const error = Math.abs(actual[j] - expected[start + j]);
        assert.ok(Number.isFinite(error), `nonfinite error at position ${position}, dim ${j}`);
        maxError = Math.max(maxError, error);
        sumSquares += error * error;
        count += 1;
    }
}

const rmsError = Math.sqrt(sumSquares / count);
console.log(JSON.stringify({
    layer: index,
    tokens: manifest.sequenceLength,
    maxAbsoluteError: maxError,
    rmsError,
}, null, 4));

// The default threshold is deliberately diagnostic, not a claim of parity.
// Supply --tolerance after observing real-checkpoint numeric behavior.
const toleranceIndex = process.argv.indexOf("--tolerance");
if (toleranceIndex >= 0) {
    const tolerance = Number(process.argv[toleranceIndex + 1]);
    assert.ok(Number.isFinite(tolerance) && tolerance >= 0, "invalid tolerance");
    assert.ok(maxError <= tolerance, `max error ${maxError} > tolerance ${tolerance}`);
    console.log("Official layer parity: PASS");
}

}
