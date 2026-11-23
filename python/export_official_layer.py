"""Export actual Gemma 3 1B weights for one-layer JS parity, no quantization.

Requires: pip install torch transformers safetensors
Checkpoint access must already be granted on Hugging Face.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"
PARAMETERS = {
    "inputLayernorm": "input_layernorm.weight",
    "postAttentionLayernorm": "post_attention_layernorm.weight",
    "preFeedforwardLayernorm": "pre_feedforward_layernorm.weight",
    "postFeedforwardLayernorm": "post_feedforward_layernorm.weight",
    "qNorm": "self_attn.q_norm.weight",
    "kNorm": "self_attn.k_norm.weight",
    "qProj": "self_attn.q_proj.weight",
    "kProj": "self_attn.k_proj.weight",
    "vProj": "self_attn.v_proj.weight",
    "oProj": "self_attn.o_proj.weight",
    "gateProj": "mlp.gate_proj.weight",
    "upProj": "mlp.up_proj.weight",
    "downProj": "mlp.down_proj.weight",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", default="fixtures/official-layer0")
    args = parser.parse_args()

    if not 0 <= args.layer < 26:
        parser.error("--layer must be between 0 and 25")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    messages = [{"role": "user", "content": "日本の首都はどこですか？"}]
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    layer = model.model.layers[args.layer]
    metadata = {"model": MODEL_ID, "layerIndex": args.layer, "tensors": {}}

    for name, path in PARAMETERS.items():
        tensor = layer
        for part in path.split("."):
            tensor = getattr(tensor, part)
        array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4")
        filename = name + ".f32"
        array.tofile(output / filename)
        metadata["tensors"][name] = {"file": filename, "shape": list(array.shape)}

    # Hidden states from the official model are captured after each layer.
    # Export *all token positions* at the input to this layer. A single final
    # token alone is insufficient to reconstruct its causal attention context.
    with torch.inference_mode():
        outputs = model(**token_ids, output_hidden_states=True, use_cache=False)

    input_states = outputs.hidden_states[args.layer][0].detach().cpu().float().numpy()
    expected_states = outputs.hidden_states[args.layer + 1][0].detach().cpu().float().numpy()
    input_states.astype("<f4").tofile(output / "input-hidden.f32")
    expected_states.astype("<f4").tofile(output / "expected-hidden.f32")

    metadata["tokenIds"] = token_ids[0].tolist()
    metadata["sequenceLength"] = len(metadata["tokenIds"])
    metadata["hiddenSize"] = input_states.shape[-1]
    metadata["inputHiddenFile"] = "input-hidden.f32"
    metadata["expectedHiddenFile"] = "expected-hidden.f32"
    metadata["referenceDtype"] = "float32"
    (output / "manifest.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(json.dumps({"output": str(output), "layer": args.layer,
                      "tokens": len(metadata["tokenIds"]),
                      "tensor_count": len(metadata["tensors"])}, indent=4))


if __name__ == "__main__":
    main()
