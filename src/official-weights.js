import { readFile } from "node:fs/promises";
import { join } from "node:path";

function float32FromBuffer(buffer, expectedLength, label) {
    if (buffer.byteLength !== expectedLength * 4) {
        throw new Error(`${label}: expected ${expectedLength * 4} bytes, got ${buffer.byteLength}`);
    }

    // Explicit little-endian decode; don't assume host endianness.
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const result = new Float32Array(expectedLength);

    for (let i = 0; i < expectedLength; i += 1) {
        result[i] = view.getFloat32(i * 4, true);
    }

    return result;
}

export async function loadOfficialLayer(directory) {
    const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"));
    const layer = {};

    for (const [name, spec] of Object.entries(manifest.tensors)) {
        const length = spec.shape.reduce((product, dimension) => product * dimension, 1);
        const buffer = await readFile(join(directory, spec.file));
        layer[name] = float32FromBuffer(buffer, length, name);
    }

    const sequenceLength = manifest.sequenceLength;
    const hiddenSize = manifest.hiddenSize;
    const length = sequenceLength * hiddenSize;
    const inputs = float32FromBuffer(await readFile(join(directory, manifest.inputHiddenFile)), length, "input-hidden");
    const expected = float32FromBuffer(await readFile(join(directory, manifest.expectedHiddenFile)), length, "expected-hidden");

    return { manifest, layer, inputs, expected };
}
