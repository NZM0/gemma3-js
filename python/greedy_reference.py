"""Generate an independent HF greedy token oracle for the Stage 05 fixed prompt."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "google/gemma-3-1b-it"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="fixtures/official-model-full")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    directory = Path(args.directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    token_ids = manifest["tokenIds"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    ids = torch.tensor([token_ids], dtype=torch.long)
    generated = []

    # No KV cache here: this is an independent, full-sequence oracle.
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
            if next_id in (1, 106):
                break

    result = {"promptTokenIds": token_ids, "generatedTokenIds": generated}
    path = directory / "greedy-reference.json"
    path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(json.dumps({"output": str(path), **result}, indent=4))


if __name__ == "__main__":
    main()
