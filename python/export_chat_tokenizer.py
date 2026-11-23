"""Export a local tokenizer for Stage 16 (requires transformers and model access)."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='fixtures/official-model')
args = parser.parse_args()
tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
backend = json.loads(tok.backend_tokenizer.to_str())
model = backend['model']
model_type = model['type']
if model_type not in ('Unigram', 'BPE'):
    raise RuntimeError(f"Unsupported tokenizer model type: {model_type}")

pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
result = {
    'model': 'google/gemma-3-1b-it',
    'tokenizerType': model_type,
    'pieces': pieces,
    'specialTokenIds': sorted(set(tok.all_special_ids)),
    'referenceText': '日本の首都はどこですか？',
    'referenceTokenIds': None,
}
# Transformers versions may return BatchEncoding rather than a plain list.
reference_encoding = tok.apply_chat_template(
    [{'role': 'user', 'content': result['referenceText']}],
    tokenize=True,
    add_generation_prompt=True,
)
if hasattr(reference_encoding, 'input_ids'):
    reference_encoding = reference_encoding['input_ids']
if hasattr(reference_encoding, 'tolist'):
    reference_encoding = reference_encoding.tolist()
if reference_encoding and isinstance(reference_encoding[0], list):
    if len(reference_encoding) != 1:
        raise RuntimeError('Expected one reference chat sequence')
    reference_encoding = reference_encoding[0]
result['referenceTokenIds'] = [int(token_id) for token_id in reference_encoding]

if model_type == 'Unigram':
    result['scores'] = [float(entry[1]) for entry in model['vocab']]
    result['unkId'] = model['unk_id']
else:
    # Preserve the official BPE configuration for a future exact JS encoder.
    # Do NOT pretend that Unigram dynamic programming can encode BPE.
    result['bpeModel'] = model
    result['normalizer'] = backend.get('normalizer')
    result['preTokenizer'] = backend.get('pre_tokenizer')
    result['decoder'] = backend.get('decoder')
    result['addedTokens'] = backend.get('added_tokens', [])

# Export independent official test vectors; the browser enables arbitrary input
# only after every relevant reference comparison succeeds.
def plain_ids(text):
    return [int(x) for x in tok.encode(text, add_special_tokens=False)]

def chat_ids(messages):
    encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, 'input_ids'):
        encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(x) for x in encoded]

samples = [
    ('japanese', '日本の首都はどこですか？'),
    ('english', 'Hello, world!'),
    ('spaces', '  A  B\nNew line'),
    ('numbers', '123 + 456 = 579'),
    ('unicode', '東京🚀 café'),
]
# Independently check the pre-tokenization boundaries, not only final IDs.
if model_type == 'BPE':
    result['preTokenizationProbes'] = [
        {
            'name': name,
            'text': value,
            'pieces': [part for part, _ in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                tok.backend_tokenizer.normalizer.normalize_str(value)
                if tok.backend_tokenizer.normalizer is not None else value
            )],
        }
        for name, value in samples
    ]

result['probes'] = [
    {'name': name, 'text': value, 'ids': plain_ids(value)}
    for name, value in samples
]
chat_samples = [
    ('single-japanese', [{'role': 'user', 'content': samples[0][1]}]),
    ('single-english', [{'role': 'user', 'content': samples[1][1]}]),
    ('two-turn', [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'},
        {'role': 'user', 'content': 'What is 2 + 2?'},
    ]),
]
result['probes'] += [
    {'name': name, 'messages': messages, 'ids': chat_ids(messages)}
    for name, messages in chat_samples
]

path = Path(args.output) / 'chat-tokenizer.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
print(f'Exported {path}, {len(result["pieces"])} pieces')
