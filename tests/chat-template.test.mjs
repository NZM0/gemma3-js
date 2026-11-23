import assert from 'node:assert/strict';
import { Gemma3ChatTokenizer } from '../src/tokenizer/gemma3-chat-tokenizer.js';

// Test the chat template independently of the BPE implementation.
const tokenizer = Object.create(Gemma3ChatTokenizer.prototype);
tokenizer.special = new Map([
    ['<bos>', 2],
    ['<start_of_turn>', 105],
    ['<end_of_turn>', 106],
]);
const textIds = new Map([
    ['user\n', [2364, 107]],
    ['model\n', [4368, 107]],
    ['日本の首都はどこですか？', [76444, 120211, 237048, 67923, 73727, 237536]],
    ['\n', [107]],
    ['Hi', [1001]],
    ['Hello!', [1002]],
]);
tokenizer.encodeText = text => {
    assert.ok(textIds.has(text), `Unexpected text: ${JSON.stringify(text)}`);
    return textIds.get(text);
};

assert.deepEqual(tokenizer.encodeMessages([
    { role: 'user', content: '日本の首都はどこですか？' },
]), [2, 105, 2364, 107, 76444, 120211, 237048, 67923, 73727, 237536, 106, 107, 105, 4368, 107]);

assert.deepEqual(tokenizer.encodeMessages([
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello!' },
]), [2, 105, 2364, 107, 1001, 106, 107, 105, 4368, 107, 1002, 106, 107, 105, 4368, 107]);

assert.throws(() => tokenizer.encodeMessages([
    { role: 'assistant', content: 'Hello!' },
]), /Expected user/);

console.log('Chat template reference tests PASS');
