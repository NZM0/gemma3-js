# gemma3-js

An independent, browser-based **Gemma 3 1B instruction-tuned text inference** implementation. Two backends are provided:

- **Native WebGPU / WGSL**, with GPU-resident transformer operations, parallel prefill and KV caching.
- **TensorFlow.js / WebGL**, an independently implemented inference path with a partitioned tied-vocabulary LM head to accommodate WebGL texture limits.

The backends share model format, tokenizer and conversation management. The application is static HTML/CSS/JavaScript; no bundler is required. This project targets the **text-only 1B model**, not the full multimodal Gemma 3 family. It is an independent implementation, not an official Google product.

## Run locally

1. Obtain the converted Gemma 3 1B IT model directory (or export one as described below). It must contain `manifest.json`, `chat-tokenizer.json`, `token-embeddings.f32`, `lm-head.f32`, `final-norm.f32`, and the `layer-00` through `layer-25` directories.
2. In this repository, run `npm run serve` (Python 3 required).
3. Open `http://localhost:8000/app/` and choose **WebGPU** or **TensorFlow.js / WebGL**.
4. Use the **local folder picker** to select the converted model directory and start chatting.

Model weights are **not** included in the source repository. The browser reads the model files from the selected local folder; the application does **not** currently implement direct Hugging Face URL loading. The TensorFlow.js demo uses TensorFlow.js 4.22.0 from a CDN, so its initial load requires network access. Browser/GPU support and available memory determine whether inference can run.

## Export model weights

The recommended source is Google's gated [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) checkpoint on Hugging Face. Accept the upstream terms and authenticate before exporting. Requires Python 3, `torch`, `numpy`, `safetensors`, and `huggingface_hub`.

```bash
python3 -m pip install torch numpy safetensors huggingface_hub
hf auth login
python3 python/export_weights.py --source hf --output ./model
python3 python/prepare_model_release.py ./model
```

The exporter requires a **new or empty output directory**. The tokenizer preparation step copies the verified BPE tokenizer from `fixtures/official-model/chat-tokenizer.json`. To reproduce an immutable source snapshot, supply `--revision <HF_COMMIT_SHA>` when exporting. The optional Google Kaggle/PyTorch checkpoint path is supported with `--source kaggle --checkpoint /path/to/model.ckpt`. Do not mix export routes in one directory.

The legacy `python/export_official_model.py` remains for diagnostic/reference fixtures; use `python/export_weights.py` for production model export. The exported `.f32` files are little-endian Float32 and **not directly compatible with Hugging Face Transformers**. They are not fine-tuned weights.

## Tests and validation

```bash
npm test
npm run test:official
```

`npm test` runs unit/structural tests and does not require GPU or full model weights. `test:official` uses separate reference fixtures and may require additional setup. Passing either is **not** a substitute for browser inference or official numerical comparison.

Validation evidence obtained during development:

- Independent Google official `gemma_pytorch` oracle comparison on a 520-token WebGPU boundary case: **3/3 greedy token IDs matched**, selected-logit maximum absolute error **6.4849854e-05** (tolerance 0.01; zero reported failures).
- Short-form WebGPU/TF.js validation, KV-cache ON/OFF checks and GPU resource-management checks were exercised in the development audit. These are not claims of exhaustive device or sequence-length coverage.
- HF-source and Kaggle-source production exports: **341/341 `.f32` files matched numerically**, **1,301,875,840 Float32 values**, zero differences; `manifest.json` JSON also matched.
- The production model was subsequently loaded locally by the project maintainer. **A separate independent Google-oracle comparison on the exact newly exported distribution package and a download/reload test from the published Hugging Face repository have not yet been recorded.**

The 512-token local-attention boundary was exercised, but full-vocabulary logit equivalence, cache ON/OFF equivalence beyond that boundary, all browser/driver combinations, and long-session stability are not claimed.

## Repository layout

- `src/backends/webgpu/`: native WebGPU implementation
- `src/backends/tfjs-webgl/`: TensorFlow.js/WebGL implementation
- `src/tokenizer/`: Gemma 3 BPE tokenizer, chat template and detokenizer
- `app/`: static chat demos and validation utilities
- `python/`: model export and reference-comparison utilities
- `fixtures/` and `tests/`: test data and regression tests

## License and upstream terms

The **original source code** in this repository is licensed under [0BSD](LICENSE). This license does **not** apply to Google's Gemma 3 model weights or tokenizer assets. Any use or redistribution of those model materials must comply with the [Gemma Terms of Use](https://gemma.ai/resources/terms/) and [Gemma Prohibited Use Policy](https://gemma.ai/resources/prohibited-use/). Converted model distributions must include the applicable agreement copy and the required `Notice` and modification disclosures. Refer to the model repository for its distribution terms and access requirements.

Gemma is a Google model; this independent project is not affiliated with or endorsed by Google.
