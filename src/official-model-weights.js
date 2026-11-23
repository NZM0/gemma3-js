import { readFile } from "node:fs/promises";
import { join } from "node:path";

export async function readF32(path, count) {
    const buffer = await readFile(path);
    if (buffer.byteLength !== count * 4) {
        throw new Error(`${path}: expected ${count * 4} bytes, got ${buffer.byteLength}`);
    }
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const output = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
        output[i] = view.getFloat32(i * 4, true);
    }
    return output;
}

export async function loadOfficialModel(directory) {
    const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"));
    const hidden = manifest.hiddenSize;
    const tokens = manifest.sequenceLength;
    const layers = [];
    for (const spec of manifest.layers) {
        const layer = {};
        for (const [name, tensor] of Object.entries(spec.tensors)) {
            const count = tensor.shape.reduce((a, b) => a * b, 1);
            layer[name] = await readF32(join(directory, spec.directory, tensor.file), count);
        }
        layers.push(layer);
    }
    const embeddings = await readF32(join(directory, manifest.embedding), tokens * hidden);
    const finalNorm = await readF32(join(directory, manifest.finalNorm), hidden);
    const referenceHidden = await readF32(join(directory, manifest.referenceHidden), (layers.length + 1) * tokens * hidden);
    const headSpec = manifest.head;
    const headRows = await readF32(join(directory, headSpec.file), headSpec.shape[0] * hidden);
    const referenceLogits = manifest.referenceLogits
        ? await readF32(join(directory, manifest.referenceLogits), headSpec.shape[0])
        : Float32Array.from(headSpec.referenceLogits);
    return { manifest, layers, embeddings, finalNorm, referenceHidden, headRows, referenceLogits };
}
