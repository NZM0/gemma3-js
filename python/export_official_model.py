"""Export full Gemma 3 1B decoder weights and prompt-specific parity oracle.

Requires: torch, transformers, numpy, safetensors and authorized model access.
Weights are NOT redistributed. Default exports only selected LM-head rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export_official_layer import MODEL_ID, PARAMETERS


def save_tensor(tensor, path):
    array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    array.tofile(path)
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fixtures/official-model")
    parser.add_argument("--full-vocab", action="store_true", help="Export full ~1.2GB LM head")
    args = parser.parse_args()
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
    if not isinstance(token_ids, torch.Tensor):
        token_ids = token_ids["input_ids"]

    pre_final_norm = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda module, args: pre_final_norm.append(args[0].detach().cpu().float())
    )
    with torch.inference_mode():
        result = model(input_ids=token_ids, output_hidden_states=True, use_cache=False)
    handle.remove()
    with torch.inference_mode():
        logits = result.logits[0, -1].float()
        top_ids = torch.topk(logits, 10).indices.tolist()
        embedding = model.model.embed_tokens(token_ids)[0]

    manifest = {
        "model": MODEL_ID,
        "tokenIds": token_ids[0].tolist(),
        "sequenceLength": token_ids.shape[1],
        "hiddenSize": embedding.shape[-1],
        "numLayers": len(model.model.layers),
        "layers": [],
        "embedding": "prompt-embeddings.f32",
        "finalNorm": "final-norm.f32",
        "referenceHidden": "reference-hidden.f32",
        "referenceTop10": top_ids,
        "head": {},
    }
    save_tensor(embedding, output / manifest["embedding"])
    save_tensor(model.model.norm.weight, output / manifest["finalNorm"])

    # All positions at every layer boundary, including the initial embedding.
    # This reference allows diagnosis of the FIRST layer where drift appears.
    # HF hidden_states[-1] is AFTER final norm; JS layer 25 output is BEFORE it.
    # Replace only the final entry with the pre-norm capture.
    states = [h[0].float().cpu() for h in result.hidden_states]
    states[-1] = pre_final_norm[0][0]
    hidden_states = torch.stack(states)
    save_tensor(hidden_states, output / manifest["referenceHidden"])

    for index, layer in enumerate(model.model.layers):
        directory = output / f"layer-{index:02d}"
        directory.mkdir(exist_ok=True)
        specs = {}
        for name, path in PARAMETERS.items():
            tensor = layer
            for part in path.split("."):
                tensor = getattr(tensor, part)
            filename = name + ".f32"
            specs[name] = {"file": filename, "shape": save_tensor(tensor, directory / filename)}
        manifest["layers"].append({"directory": directory.name, "tensors": specs})
        print(f"Exported layer {index + 1}/{len(model.model.layers)}", flush=True)

    # Gemma 3 ties LM head to input embeddings. Export either all rows or a
    # diagnostic subset; subset mode cannot prove global top-k equivalence.
    head = model.get_output_embeddings().weight
    if args.full_vocab:
        filename = "lm-head.f32"
        manifest["head"] = {
            "mode": "full", "file": filename, "shape": save_tensor(head, output / filename)
        }
        save_tensor(logits, output / "reference-logits.f32")
        manifest["referenceLogits"] = "reference-logits.f32"
    else:
        # Top-10 reference candidates plus stable non-top-10 controls.
        candidate_ids = sorted(set(top_ids + [0, 1, 2, 10, 100, 1000, 10000]))
        selected = head[candidate_ids]
        filename = "lm-head-candidates.f32"
        manifest["head"] = {
            "mode": "candidates", "file": filename,
            "shape": save_tensor(selected, output / filename),
            "tokenIds": candidate_ids,
            "referenceLogits": logits[candidate_ids].tolist(),
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "layers": len(manifest["layers"]),
        "tokens": token_ids.shape[1], "headMode": manifest["head"]["mode"],
        "referenceTop10": top_ids,
    }, indent=4))


if __name__ == "__main__":
    main()
