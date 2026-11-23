#!/usr/bin/env python3
"""Export Gemma 3 1B IT weights to gemma3-js f32 manifest format.

HF path is primary. Kaggle Google gemma_pytorch checkpoint is optional.
No inference, diagnostic hidden states or synthetic embeddings are emitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

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


def write_tensor(tensor, path):
    array = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    array.astype("<f4", copy=False).tofile(path)
    return list(array.shape)


def export_hf(output, revision):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    source = Path(snapshot_download(repo_id=MODEL_ID, revision=revision,
                                    allow_patterns=["*.safetensors", "*.json"]))
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("No safetensors files found")
    mapping = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in mapping:
                    raise ValueError(f"Duplicate key: {key}")
                mapping[key] = shard

    def get(key):
        if key not in mapping:
            raise KeyError(key)
        with safe_open(str(mapping[key]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    return get, {"source": "huggingface", "model": MODEL_ID,
                 "revision": revision or "default branch (pin before publishing)"}


def export_kaggle(checkpoint):
    raw = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = raw.get("model_state_dict", raw)
    if not isinstance(state, dict):
        raise ValueError("Expected a PyTorch state dictionary")

    def get(key):
        if key.startswith("model.layers."):
            parts = key.split(".")
            index = parts[2]
            suffix = ".".join(parts[3:])
            prefix = f"model.layers.{index}."
            if suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight"):
                combined = state[prefix + "self_attn.qkv_proj.weight"]
                q, k, v = combined.split([1024, 256, 256], dim=0)
                return {"self_attn.q_proj.weight": q,
                        "self_attn.k_proj.weight": k,
                        "self_attn.v_proj.weight": v}[suffix]
            suffix = suffix.replace("self_attn.q_norm.", "self_attn.query_norm.")
            suffix = suffix.replace("self_attn.k_norm.", "self_attn.key_norm.")
            return state[prefix + suffix]
        if key == "model.embed_tokens.weight":
            return state["embedder.weight"]
        return state[key]

    return get, {"source": "kaggle", "checkpoint": str(Path(checkpoint).resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    parser.add_argument("--checkpoint", help="Kaggle model.ckpt for --source kaggle")
    parser.add_argument("--revision", help="HF immutable commit SHA (recommended)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.source == "kaggle" and not args.checkpoint:
        parser.error("--checkpoint is required for Kaggle")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty (avoid mixing versions)")
    output.mkdir(parents=True, exist_ok=True)
    get, provenance = (export_hf(output, args.revision) if args.source == "hf"
                       else export_kaggle(args.checkpoint))

    embedding = get("model.embed_tokens.weight")
    if list(embedding.shape) != [262144, 1152]:
        raise ValueError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    manifest = {
        "model": MODEL_ID,
        "hiddenSize": 1152,
        "numLayers": 26,
        "layers": [],
        "tokenEmbedding": "token-embeddings.f32",
        "finalNorm": "final-norm.f32",
        "head": {"mode": "full", "file": "lm-head.f32", "shape": [262144, 1152]},
    }
    write_tensor(embedding, output / "token-embeddings.f32")
    # Tied weights: duplicate for compatibility with existing LM head filename.
    write_tensor(embedding, output / "lm-head.f32")
    del embedding
    write_tensor(get("model.norm.weight"), output / "final-norm.f32")
    for index in range(26):
        directory = output / f"layer-{index:02d}"
        directory.mkdir()
        tensors = {}
        for name, suffix in PARAMETERS.items():
            weight = get(f"model.layers.{index}.{suffix}")
            tensors[name] = {"file": name + ".f32",
                             "shape": write_tensor(weight, directory / (name + ".f32"))}
            del weight
        manifest["layers"].append({"directory": directory.name, "tensors": tensors})
        print(f"Exported layer {index + 1}/26", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=4) + "\n")
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=4) + "\n")
    print("Export finished. This manifest is the production model format;"
          " legacy diagnostic loaders may require prompt-specific fields."
          " Do not publish until a real inference loader and oracle comparison pass.")


if __name__ == "__main__":
    main()
