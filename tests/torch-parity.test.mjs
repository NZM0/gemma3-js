import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DecoderSession, decoderStep } from "../src/gemma3-decoder.js";
import { SessionCache } from "../src/gemma3-core.js";

const fixture = JSON.parse(readFileSync(new URL("../fixtures/torch-decoder-reference.json", import.meta.url), "utf8"));
const layers = fixture.layers.map(layer => Object.fromEntries(
    Object.entries(layer).map(([name, values]) => [name, Float32Array.from(values)])
));
const session = new DecoderSession(fixture.config, layers);
const tolerance = 3e-5;
let maximumError = 0;

function compare(actual, expected, label) {
    assert.equal(actual.length, expected.length, `${label}: length`);

    for (let i = 0; i < actual.length; i += 1) {
        const error = Math.abs(actual[i] - expected[i]);
        assert.ok(Number.isFinite(error), `${label}[${i}]: nonfinite`);
        maximumError = Math.max(maximumError, error);
        assert.ok(error < tolerance, `${label}[${i}]: ${actual[i]} vs ${expected[i]}, error ${error}`);
    }
}

// Inspect each layer independently, with its own state, as well as the session.
const caches = layers.map(() => new SessionCache(layers.length));

for (let position = 0; position < fixture.inputs.length; position += 1) {
    let hidden = Float32Array.from(fixture.inputs[position]);

    for (let index = 0; index < layers.length; index += 1) {
        hidden = decoderStep(hidden, layers[index], caches[index], index, position, fixture.config);
        compare(hidden, fixture.expected[position].layerOutputs[index], `position ${position}, layer ${index}`);
    }

    const actual = session.step(fixture.inputs[position]);
    compare(actual, fixture.expected[position].output, `position ${position}, final`);
}

assert.deepEqual(session.cache.layers[0].positions, [3, 4]);
assert.deepEqual(session.cache.layers[1].positions, [0, 1, 2, 3, 4]);
console.log(`Gemma 3 PyTorch synthetic parity: PASS (max error ${maximumError}, tolerance ${tolerance})`);
