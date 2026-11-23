import assert from "node:assert/strict";
import { Gemma3Detokenizer } from "../src/tokenizer/gemma3-detokenizer.js";

const decoder = new Gemma3Detokenizer([
    "<pad>", "▁こんにちは", "世界", "▁Gemma", "!", "<0xE3>", "<0x81>", "<0x82>", "<eos>", "▁"
], [0, 8]);

assert.equal(decoder.decode([1, 2, 3, 4, 8]), "こんにちは世界 Gemma!");
assert.equal(decoder.decode([5, 6, 7]), "あ");
assert.equal(decoder.decode([1, 9, 2]), "こんにちは 世界");
assert.equal(decoder.decode([1, 8], { skipSpecialTokens: false }), "こんにちは<eos>");
assert.throws(() => decoder.decode([99]), /Invalid token ID/);
assert.throws(() => decoder.decode([5]), /encoded data|UTF-8|valid/i);
// Gemma's end-of-turn marker may be absent from all_special_ids.
const gemmaChat = new Gemma3Detokenizer([
    "<pad>", "▁日本の首都は東京です。", "<end_of_turn>", "<start_of_turn>"
], [0]);
assert.equal(gemmaChat.decode([1, 2]), "日本の首都は東京です。");
assert.equal(gemmaChat.decode([1, 2], { skipSpecialTokens: false }), "日本の首都は東京です。<end_of_turn>");
assert.equal(gemmaChat.decode([3, 1, 2]), "日本の首都は東京です。");
console.log("Gemma 3 detokenizer: synthetic tests passed");
