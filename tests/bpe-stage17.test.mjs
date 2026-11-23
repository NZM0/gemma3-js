import assert from 'node:assert/strict';
import { Gemma3Bpe } from '../src/tokenizer/gemma3-bpe.js';

const data = {
    bpeModel: {
        type: 'BPE',
        vocab: { '▁': 0, H: 1, e: 2, l: 3, o: 4, 'He': 5, 'Hel': 6, 'Hell': 7, 'Hello': 8, '▁Hello': 9 },
        merges: ['H e', 'He l', 'Hel l', 'Hell o', '▁ Hello'],
        unk_token: null,
        byte_fallback: false,
        dropout: null,
    },
    preTokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always' },
    normalizer: null,
};
const tokenizer = new Gemma3Bpe(data);
assert.deepEqual(tokenizer.encode('Hello'), [9]);
assert.throws(() => new Gemma3Bpe({ ...data, preTokenizer: { type: 'ByteLevel' } }), /Unsupported/);
console.log('Stage 17 BPE basic rank merge test: PASS');

const splitData = {
    ...data,
    preTokenizer: {
        type: 'Split',
        pattern: { String: ' ' },
        behavior: 'MergedWithPrevious',
        invert: false,
    },
    bpeModel: {
        ...data.bpeModel,
        vocab: { H: 0, e: 1, l: 2, o: 3, ' ': 4, w: 5, r: 6, d: 7, 'He': 8, 'Hel': 9, 'Hell': 10, 'Hello': 11, 'Hello ': 12, 'wo': 13, 'wor': 14, 'worl': 15, 'world': 16 },
        merges: ['H e', 'He l', 'Hel l', 'Hell o', ['Hello', ' '], 'w o', 'wo r', 'wor l', 'worl d'],
    },
};
const split = new Gemma3Bpe(splitData);
assert.deepEqual(split.preTokenize('Hello world'), ['Hello ', 'world']);
assert.deepEqual(split.preTokenize('  A  B\nNew line'), [' ', ' ', 'A ', ' ', 'B\nNew ', 'line']);
assert.deepEqual(split.encode('Hello world'), [12, 16]);
assert.throws(() => new Gemma3Bpe({ ...splitData, preTokenizer: { ...splitData.preTokenizer, behavior: 'Isolated' } }), /Unsupported/);
console.log('Stage 17 Split MergedWithPrevious test: PASS');

const normalizedData = {
    ...splitData,
    normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
};
const normalized = new Gemma3Bpe(normalizedData);
assert.equal(normalized.normalize('Hello  world'), 'Hello▁▁world');
assert.deepEqual(normalized.preTokenize('Hello world'), ['Hello▁world']);
assert.throws(() => new Gemma3Bpe({ ...splitData, normalizer: { type: 'Replace', pattern: { Regex: ' +' }, content: '▁' } }), /Unsupported/);
console.log('Stage 17 literal Replace normalizer test: PASS');
