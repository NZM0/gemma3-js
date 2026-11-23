"""Export a genuine 520+ token PyTorch oracle for the existing Stage 05 weights.

The model's official 512-token local window is not modified. No weights are
included in the reference export. Requires HF checkpoint access.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export_official_layer import MODEL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fixtures/long-reference")
    parser.add_argument("--length", type=int, default=520)
    parser.add_argument("--text", default="日本の首都は東京です。日本には多くの都市があります。")
    args = parser.parse_args()
    if args.length < 513 or not args.text:
        parser.error("--length must be >= 513 and --text must not be empty")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    # Use a real chat template, then extend the user's text until the prompt
    # reaches the target length. This is NOT a repetition of hidden states.
    messages = [{"role": "user", "content": args.text}]
    while True:
        encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = encoding if isinstance(encoding, torch.Tensor) else encoding["input_ids"]
        if ids.shape[1] >= args.length:
            break
        messages[0]["content"] += " " + args.text

    ids = ids[:, :args.length].contiguous()
    # The final tokens may be a truncated user message; that's intentional:
    # this test verifies forward numerical parity, not chat completion quality.
    captured = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().float())
    )
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()

    embedding = model.model.embed_tokens(ids)[0].detach().cpu().float()
    states = [state[0].detach().cpu().float() for state in result.hidden_states]
    states[-1] = captured[0][0]
    checkpoints = sorted(set([0, 1, 510, 511, 512, 513, args.length - 1]))
    checkpoints = [p for p in checkpoints if p < args.length]

    # Save only selected positions for each layer, in [layer, checkpoint, hidden]
    # order. The JS runner compares every decoder layer at these positions.
    selected = torch.stack([torch.stack([s[p] for p in checkpoints]) for s in states])
    embedding.numpy().astype("<f4").tofile(output / "embeddings.f32")
    selected.numpy().astype("<f4").tofile(output / "checkpoint-hidden.f32")
    last_logits = result.logits[0, -1].detach().cpu().float()
    top10 = torch.topk(last_logits, 10).indices.tolist()
    last_logits.numpy().astype("<f4").tofile(output / "last-logits.f32")

    metadata = {
        "model": MODEL_ID,
        "length": args.length,
        "hiddenSize": embedding.shape[-1],
        "numLayers": len(model.model.layers),
        "vocabSize": last_logits.numel(),
        "checkpoints": checkpoints,
        "tokenIds": ids[0].tolist(),
        "embeddingFile": "embeddings.f32",
        "hiddenFile": "checkpoint-hidden.f32",
        "logitsFile": "last-logits.f32",
        "referenceTop10": top10,
        "windowSize": 512,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "length": args.length,
        "checkpoints": checkpoints, "referenceTop10": top10,
    }, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
