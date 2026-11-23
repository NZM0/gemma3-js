import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { GEMMA3_1B } from "../../src/gemma3-core.js";
import { Gemma3Generator } from "../../src/gemma3-generate.js";
import { Gemma3Detokenizer } from "../../src/tokenizer/gemma3-detokenizer.js";
import { compareCachedAndRecomputed } from "../../src/gemma3-cache-parity.js";
import { loadOfficialModel } from "../../src/official-model-weights.js";

const directory = process.argv[2] ?? "fixtures/official-model-full";
const maxNewTokens = Number(process.argv[3] ?? 3);
const tolerance = Number(process.argv[4] ?? 1e-5);

if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || !Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error("Usage: node tests/official-cache-parity.mjs DIRECTORY [maxNewTokens] [tolerance]");
}

if (!existsSync(join(directory, "manifest.json"))) {
    throw new Error("Missing model export; use the Stage 07 --full-vocab export");
}

const data = await loadOfficialModel(directory);

if (data.manifest.head.mode !== "full") {
    throw new Error("Cache parity requires a --full-vocab model export");
}

const config = { ...GEMMA3_1B, numLayers: data.layers.length };
const generator = new Gemma3Generator(data.layers, data.finalNorm, data.headRows, config);
const promptIds = data.manifest.tokenIds;

console.log(`Cache parity: ${promptIds.length} prompt tokens; up to ${maxNewTokens} generated tokens`);
console.log("Recompute mode replays the entire prefix at each generation step; it may be slow on CPU.");

const result = compareCachedAndRecomputed(generator, promptIds, maxNewTokens, tolerance);
const piecesPath = join(directory, "tokenizer-pieces.json");

if (existsSync(piecesPath)) {
    const vocabulary = JSON.parse(await readFile(piecesPath, "utf8"));
    const detokenizer = new Gemma3Detokenizer(vocabulary.pieces, vocabulary.specialTokenIds);
    result.decodedText = detokenizer.decode(result.generatedTokenIds);
}

const oraclePath = join(directory, "greedy-reference.json");

if (existsSync(oraclePath)) {
    const reference = JSON.parse(await readFile(oraclePath, "utf8"));
    const expected = reference.generatedTokenIds.slice(0, result.generatedTokenIds.length);
    result.matchesOfficialReference = result.generatedTokenIds.length === Math.min(maxNewTokens, reference.generatedTokenIds.length)
        && result.generatedTokenIds.every((tokenId, i) => tokenId === expected[i]);
}

console.log(JSON.stringify(result, null, 4));

if (!result.matches || result.matchesOfficialReference === false) {
    process.exitCode = 1;
}
