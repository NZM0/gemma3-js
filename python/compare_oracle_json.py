"""Strict independent oracle comparison. Empty or incomplete traces always fail."""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_trace(trace):
    require(trace.get('schema') == 'gemma3-js-parity-trace-v1', 'Unsupported trace schema')
    turns = trace.get('turns')
    require(isinstance(turns, list) and len(turns) > 0, 'Trace has no turns: export after generating a reply')
    for i, turn in enumerate(turns, 1):
        prompt = turn.get('promptTokenIds')
        tokens = turn.get('generatedTokenIds')
        logits = turn.get('selectedLogits')
        require(isinstance(prompt, list) and len(prompt) > 0, f'Turn {i}: empty prompt')
        require(isinstance(tokens, list) and len(tokens) > 0, f'Turn {i}: no generated tokens')
        require(isinstance(logits, list) and len(logits) == len(tokens), f'Turn {i}: missing selected logits')
        require(all(isinstance(x, int) and x >= 0 for x in prompt + tokens), f'Turn {i}: invalid token ID')
        require(all(isinstance(x, (float, int)) and math.isfinite(x) for x in logits), f'Turn {i}: invalid logit')
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('oracle', type=Path)
    parser.add_argument('--tolerance', type=float, default=0.01)
    args = parser.parse_args()
    require(math.isfinite(args.tolerance) and args.tolerance >= 0, 'Invalid tolerance')
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    oracle = json.loads(args.oracle.read_text(encoding='utf-8'))
    turns = validate_trace(trace)
    require(oracle.get('schema') == 'gemma3-parity-oracle-v1', 'Unsupported oracle schema')
    require(len(turns) == len(oracle.get('turns', [])), 'Turn count differs')
    failures = 0
    max_diff = 0.0
    checked = 0
    for ti, (turn, reference) in enumerate(zip(turns, oracle['turns']), 1):
        require(turn['promptTokenIds'] == reference.get('promptTokenIds'), f'Turn {ti}: prompt token IDs differ')
        steps = reference.get('steps')
        require(isinstance(steps, list) and len(steps) == len(turn['generatedTokenIds']), f'Turn {ti}: step count differs')
        for si, (token, logit, step) in enumerate(zip(turn['generatedTokenIds'], turn['selectedLogits'], steps), 1):
            require(step.get('selectedTokenId') == token, f'Turn {ti}, step {si}: oracle selected-token ID mismatch')
            oracle_logit = step.get('selectedLogit')
            top1_logit = step.get('top1Logit')
            require(isinstance(oracle_logit, (float, int)) and math.isfinite(oracle_logit), f'Turn {ti}, step {si}: invalid oracle logit')
            require(isinstance(top1_logit, (float, int)) and math.isfinite(top1_logit), f'Turn {ti}, step {si}: invalid oracle top1 logit')
            require(isinstance(step.get('tokenId'), int), f'Turn {ti}, step {si}: invalid oracle top1 ID')
            diff = abs(logit - oracle_logit)
            max_diff = max(max_diff, diff)
            matched = token == step['tokenId'] and diff <= args.tolerance
            failures += not matched
            checked += 1
            print(f'Turn {ti}, token {si}: JS={token}, oracle top1={step["tokenId"]}, '
                  f'selected-logit abs diff={diff:.8g}, {"PASS" if matched else "FAIL"}')
    require(checked > 0, 'No comparison steps executed')
    print(f'Provider: {oracle.get("provider", "unknown")}')
    print(f'Compared steps: {checked}; max selected-logit error: {max_diff:.8g}; failures: {failures}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
