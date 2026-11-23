"""Numerical comparison of exported gemma3-js traces against HF Transformers.

HF is a secondary independent numerical oracle, NOT Google's gemma_pytorch.
For the primary Google-official parity claim, run an equivalent trace with
Google's gemma_pytorch implementation and compare using compare_oracle_json.py.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--model', default='google/gemma-3-1b-it')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--output', type=Path, default=Path('gemma3-reference-hf.json'))
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    from compare_oracle_json import validate_trace
    turns = validate_trace(trace)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True).eval()
    oracle = {'schema': 'gemma3-parity-oracle-v1', 'provider': 'Hugging Face Transformers (secondary)',
              'model': args.model, 'dtype': args.dtype, 'turns': []}

    with torch.inference_mode():
        for turn_index, turn in enumerate(turns):
            ids = list(turn['promptTokenIds'])
            expected = []
            for step, selected_id in enumerate(turn['generatedTokenIds']):
                logits = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False).logits[0, -1].float()
                expected.append({'tokenId': int(torch.argmax(logits).item()),
                                 'selectedTokenId': int(selected_id),
                                 'selectedLogit': float(logits[selected_id].item()),
                                 'top1Logit': float(logits.max().item())})
                ids.append(selected_id)  # Teacher forcing: same prefix as JS, even if greedy diverges
            oracle['turns'].append({'promptTokenIds': turn['promptTokenIds'], 'steps': expected})
            print(f'Turn {turn_index + 1}: {len(expected)} reference steps')

    if not oracle['turns'] or not any(t['steps'] for t in oracle['turns']):
        raise ValueError('No reference logits computed')
    args.output.write_text(json.dumps(oracle, ensure_ascii=False, indent=4), encoding='utf-8')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
