"""Capture intermediate hidden states from the real official Gemma 3 1B IT checkpoint.

Requires torch, transformers, Hugging Face checkpoint access. This is an
independent official-model baseline; Stage 03 does not yet load these weights
in JS, so official JS parity is NOT claimed.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"
OUTPUT = Path(__file__).resolve().parents[1] / "fixtures" / "official-trace.json"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    messages = [{"role": "user", "content": "日本の首都はどこですか？"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(input_ids=inputs, output_hidden_states=True, use_cache=False)

    logits = outputs.logits[0, -1]
    values, indices = torch.topk(logits, 10)
    result = {
        "model": MODEL_ID,
        "dtype": "float32",
        "token_ids": inputs[0].tolist(),
        "hidden_states": [state[0, -1].tolist() for state in outputs.hidden_states],
        "top10_token_ids": indices.tolist(),
        "top10_logits": values.tolist(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "tokens": len(result["token_ids"]),
        "hidden_state_count": len(result["hidden_states"]),
        "hidden_size": len(result["hidden_states"][0]),
        "top10_token_ids": result["top10_token_ids"],
    }, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
