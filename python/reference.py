"""Official-checkpoint reference probe; requires local torch + transformers and HF access."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"


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

    with torch.no_grad():
        logits = model(inputs).logits[0, -1]

    values, indices = torch.topk(logits, 10)
    print(json.dumps({
        "model": MODEL_ID,
        "token_ids": inputs[0].tolist(),
        "top10_token_ids": indices.tolist(),
        "top10_logits": values.tolist(),
    }, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
