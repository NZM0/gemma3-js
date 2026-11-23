#!/usr/bin/env python3
"""Copy the validated chat tokenizer into an exported model directory."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("fixtures/official-model/chat-tokenizer.json"))
    args = parser.parse_args()

    manifest_path = args.model_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("head", {}).get("mode") != "full":
        parser.error("Expected a full-vocabulary LM head")

    data = json.loads(args.tokenizer.read_text())
    if data.get("model") != manifest.get("model"):
        parser.error("Tokenizer and manifest model identifiers differ")
    if data.get("tokenizerType") != "BPE" or not data.get("probes"):
        parser.error("Expected BPE tokenizer with verification probes")

    target = args.model_dir / "chat-tokenizer.json"
    if args.tokenizer.resolve() != target.resolve():
        shutil.copyfile(args.tokenizer, target)
    print(f"Tokenizer ready: {target}")
    print("This checks package structure only; browser inference still requires validation.")


if __name__ == "__main__":
    main()
