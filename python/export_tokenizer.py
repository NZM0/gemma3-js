"""Export official Gemma token pieces and reference decoded output.

Requires: pip install transformers sentencepiece
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="fixtures/official-model-full")
    args = parser.parse_args()

    directory = Path(args.directory)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    if any(not isinstance(piece, str) for piece in pieces):
        raise RuntimeError("Tokenizer contains an ID without a string piece")

    # Some Gemma chat delimiters are not reported by all_special_ids.
    chat_markers = {"<start_of_turn>", "<end_of_turn>", "<|im_start|>", "<|im_end|>"}
    special_ids = sorted(
        set(tokenizer.all_special_ids)
        | {i for i, piece in enumerate(pieces) if piece in chat_markers}
    )
    result = {
        "model": MODEL_ID,
        "pieces": pieces,
        "specialTokenIds": special_ids,
    }
    path = directory / "tokenizer-pieces.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    oracle_path = directory / "greedy-reference.json"
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        generated = oracle["generatedTokenIds"]
        oracle["referenceDecodedText"] = tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        oracle_path.write_text(json.dumps(oracle, ensure_ascii=False, indent=4), encoding="utf-8")
        print(json.dumps({"referenceDecodedText": oracle["referenceDecodedText"]}, ensure_ascii=False))

    print(json.dumps({"output": str(path), "vocabSize": len(pieces), "specialTokens": len(special_ids)}))


if __name__ == "__main__":
    main()
