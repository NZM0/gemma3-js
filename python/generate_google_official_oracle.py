"""Google gemma_pytorch independent oracle for gemma3-js parity traces.

Uses Google's GemmaForCausalLM.forward, not a reimplementation of its layers.
Teacher-forces the recorded token IDs; each turn starts with an empty KV cache.
"""
import argparse
import contextlib
import json
import math
import sys
from pathlib import Path


def validate(trace):
    if trace.get('schema') != 'gemma3-js-parity-trace-v1':
        raise ValueError('Unsupported trace schema')
    if trace.get('model') != 'google/gemma-3-1b-it':
        raise ValueError('This oracle supports only google/gemma-3-1b-it')
    turns = trace.get('turns')
    if not isinstance(turns, list) or not turns:
        raise ValueError('No turns in trace')
    for number, turn in enumerate(turns, 1):
        prompt = turn.get('promptTokenIds')
        generated = turn.get('generatedTokenIds')
        logits = turn.get('selectedLogits')
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f'Turn {number}: missing promptTokenIds')
        if not isinstance(generated, list) or not generated:
            raise ValueError(f'Turn {number}: missing generatedTokenIds')
        if not isinstance(logits, list) or len(logits) != len(generated):
            raise ValueError(f'Turn {number}: missing selectedLogits')
        if any(type(token) is not int or not 0 <= token < 262144 for token in prompt + generated):
            raise ValueError(f'Turn {number}: invalid token ID')
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in logits):
            raise ValueError(f'Turn {number}: invalid selectedLogit')
    return turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--google-repo', type=Path, required=True, help='Cloned google/gemma_pytorch repository')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Official PyTorch .ckpt or sharded PyTorch checkpoint directory')
    parser.add_argument('--tokenizer', type=Path, required=True, help='Gemma 3 SentencePiece tokenizer model file')
    parser.add_argument('--output', type=Path, default=Path('gemma3-reference-google.json'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--turn', type=int, default=None, help='Optional 1-based turn to evaluate; writes all turns only when omitted')
    args = parser.parse_args()

    if not args.google_repo.is_dir() or not (args.google_repo / 'gemma' / 'model.py').is_file():
        parser.error('--google-repo must contain gemma/model.py')
    if not args.checkpoint.exists() or not args.tokenizer.is_file():
        parser.error('Checkpoint and tokenizer must exist')
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    turns = validate(trace)
    if args.turn is not None and not 1 <= args.turn <= len(turns):
        parser.error('--turn is out of range')

    sys.path.insert(0, str(args.google_repo.resolve()))
    import torch
    from gemma import config as gemma_config
    from gemma import model as gemma_model

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    config = gemma_config.get_config_for_1b(args.dtype)
    config.tokenizer = str(args.tokenizer.resolve())

    # The official model initializes parameters with the current default dtype.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = gemma_model.GemmaForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)

    model.load_weights(str(args.checkpoint.resolve()))
    model = model.to(device).eval()
    print('Google official checkpoint loaded', flush=True)

    results = []
    with torch.inference_mode():
        for turn_index, turn in enumerate(turns, 1):
            if args.turn is not None and args.turn != turn_index:
                continue
            prompt = turn['promptTokenIds']
            generated = turn['generatedTokenIds']
            total_length = len(prompt) + len(generated) - 1
            if total_length > config.max_position_embeddings:
                raise ValueError(f'Turn {turn_index}: exceeds maximum position embeddings')

            # Allocate cache for this turn, including all prompt and generated positions.
            shape = (1, total_length, config.num_key_value_heads, config.head_dim)
            caches = [
                (torch.zeros(shape, dtype=dtype, device=device),
                 torch.zeros(shape, dtype=dtype, device=device))
                for _ in range(config.num_hidden_layers)
            ]
            steps = []
            tokens = prompt + generated[:-1]

            # Single-token forwards avoid ambiguous batched prefill semantics
            # and use Google's own absolute-position cache and local masks.
            for position, input_id in enumerate(tokens):
                positions = torch.tensor([position], dtype=torch.long, device=device)
                ids = torch.tensor([[input_id]], dtype=torch.long, device=device)
                keys = torch.arange(total_length, device=device)
                global_invalid = keys > position
                local_invalid = global_invalid | (keys <= position - config.sliding_window_size)
                negative = -2.3819763e38
                global_mask = torch.where(global_invalid, negative, 0.0).to(torch.float32).view(1, 1, 1, -1)
                local_mask = torch.where(local_invalid, negative, 0.0).to(torch.float32).view(1, 1, 1, -1)

                _, logits = model(
                    input_token_ids=ids,
                    input_positions=positions,
                    kv_write_indices=positions,
                    kv_caches=caches,
                    mask=global_mask,
                    output_positions=torch.tensor([0], dtype=torch.long, device=device),
                    temperatures=None,
                    top_ps=torch.tensor([1.0], device=device),
                    top_ks=torch.tensor([1], dtype=torch.long, device=device),
                    local_mask=local_mask,
                )

                if position >= len(prompt) - 1:
                    step_index = position - (len(prompt) - 1)
                    selected = generated[step_index]
                    row = logits[0].float()
                    steps.append({
                        'selectedTokenId': selected,
                        'selectedLogit': float(row[selected].item()),
                        'tokenId': int(row.argmax().item()),
                        'top1Logit': float(row.max().item()),
                    })
            results.append({'promptTokenIds': prompt, 'steps': steps})
            print(f'Turn {turn_index}: {len(steps)} oracle steps; last position {total_length - 1}', flush=True)
            del caches

    if args.turn is not None:
        # A partial oracle is intentionally incompatible with the strict
        # all-turn comparator; use a single-turn trace for comparison.
        output_turns = results
    else:
        output_turns = results
    output = {
        'schema': 'gemma3-parity-oracle-v1',
        'provider': 'Google gemma_pytorch (official)',
        'model': trace['model'],
        'dtype': args.dtype,
        'turns': output_turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
