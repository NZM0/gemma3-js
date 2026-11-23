import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { GEMMA3_1B } from "../../src/gemma3-core.js";
import { Gemma3Generator } from "../../src/gemma3-generate.js";
import { Gemma3Detokenizer } from "../../src/tokenizer/gemma3-detokenizer.js";
import { loadOfficialModel } from "../../src/official-model-weights.js";

const directory = process.argv[2] ?? "fixtures/official-model-full";
const maxNewTokens = Number(process.argv[3] ?? 8);
if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1) throw new Error("Invalid maxNewTokens");
if (!existsSync(join(directory, "manifest.json"))) {
    throw new Error("Missing export. Run export_official_model.py --full-vocab first.");
}
const data = await loadOfficialModel(directory);
if (data.manifest.head.mode !== "full") {
    throw new Error("Greedy generation needs --full-vocab export (tied embedding/LM head)");
}
const config = { ...GEMMA3_1B, numLayers: data.layers.length };
const generator = new Gemma3Generator(data.layers, data.finalNorm, data.headRows, config);
const tokenIds = data.manifest.tokenIds;
console.log(`Prefill: ${tokenIds.length} tokens; generating up to ${maxNewTokens} tokens`);
const generated = generator.generate(tokenIds, maxNewTokens, (id, step, logit) => {
    console.log(`step=${step + 1} token=${id} logit=${logit}`);
});
const output = { promptTokenIds: tokenIds, generatedTokenIds: generated };
const piecesPath = join(directory, "tokenizer-pieces.json");
if (existsSync(piecesPath)) {
    const vocabulary = JSON.parse(await readFile(piecesPath, "utf8"));
    const detokenizer = new Gemma3Detokenizer(vocabulary.pieces, vocabulary.specialTokenIds);
    output.decodedText = detokenizer.decode(generated);
    console.log(`Generated text: ${output.decodedText}`);
} else {
    console.log("No tokenizer-pieces.json found; run python3 python/export_tokenizer.py first.");
}
const oraclePath = join(directory, "greedy-reference.json");
if (existsSync(oraclePath)) {
    const reference = JSON.parse(await readFile(oraclePath, "utf8"));
    const expected = reference.generatedTokenIds.slice(0, generated.length);
    const matches = generated.every((id, index) => id === expected[index]);
    if (typeof reference.referenceDecodedText === "string" && output.decodedText !== undefined) {
        output.referenceDecodedText = reference.referenceDecodedText;
        output.matchesDecodedText = output.decodedText === reference.referenceDecodedText;
        if (!output.matchesDecodedText) process.exitCode = 1;
    }
    output.referenceTokenIds = expected;
    output.matchesReference = matches && generated.length === Math.min(maxNewTokens, reference.generatedTokenIds.length);
    if (!output.matchesReference) process.exitCode = 1;
}
console.log(JSON.stringify(output, null, 4));
