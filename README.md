# gemma3-js

**v0.3.0 — WGSL Turbo**

An independent, browser-based **Gemma 3 1B instruction-tuned text inference** implementation. Three main backends are provided:

- **WGSL Turbo**, an optimized native WebGPU backend with no additional quantization or runtime dependencies.
- **Native WebGPU / WGSL Reference**, with GPU-resident transformer operations, parallel prefill and KV caching.
- **TensorFlow.js / WebGL**, an independently implemented inference path with a partitioned tied-vocabulary LM head to accommodate WebGL texture limits.

The backends share model format, tokenizer and conversation management. The application is static HTML/CSS/JavaScript; no bundler is required. This project targets the **text-only 1B model**, not the full multimodal Gemma 3 family. It is an independent implementation, not an official Google product.

## WGSL Turbo performance

Same Mobile INT4 weights, prompt `Where is the capital of Japan?`, 16 input tokens, 24 generated tokens, KV cache enabled. Individual maintainer measurements, not controlled averages.

| Device | Reference decode (tok/s) | Turbo decode (tok/s) | Speedup | Turbo prefill (ms) | Turbo TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mac M4 | 10.11 | 57.31 | 5.67x | 161.6 | 167.8 |
| iPhone 16e | 3.90 | 13.58 | 3.48x | 637 | 661 |

Mac Turbo repeat runs: 54.02 and 54.12 tok/s. The iPhone Reference baseline above used remote model loading; Turbo used bundled model files served over local HTTPS. Inference ran on the device in both cases. The historical local Reference result was 3.88 tok/s.

Tracked GPU buffers on Mac: 877.8 MiB (Reference) to 637.8 MiB (Turbo), a 240 MiB reduction. These are not total browser memory measurements.

Turbo uses parallel reductions, tiled online-softmax attention, parallel INT4 matrix-vector operations, tiled prefill, shared scratch buffers and dynamically growing global KV caches. No model reconversion is required. See [WGSL-TURBO.md](WGSL-TURBO.md) for validation scope and limitations.

[Open Turbo](app/webgpu-turbo/) · [Changelog](CHANGELOG.md)

## Run locally

1. Obtain the converted Gemma 3 1B IT model directory (or export one as described below). It must contain `manifest.json`, `chat-tokenizer.json`, `lm-head.f32`, `final-norm.f32`, and the `layer-00` through `layer-25` directories.
2. In this repository, run `npm run serve` (Python 3 required).
3. Open `http://localhost:8000/app/` and choose **WGSL Turbo**, **WebGPU Reference** or **TensorFlow.js / WebGL**.
4. Select the local folder or Hugging Face model source, then load the model.

Model weights and Google-derived tokenizer data are **not** included in the source repository. Select Local folder or Hugging Face in the applicable chat demo. The remote source uses the public NZM0/gemma3-js-1b-it repository and requires HTTPS, CORS and exact HTTP 206 byte ranges for sliced reads. WebGPU uploads the head in chunks and reads remote-mode embeddings from resident GPU weights; WebGL still loads the full head and uses remote row reads. New production exports store tied weights once; older exports remain supported. Remote FP32 loading and generation were exercised on the maintainer’s Mac using both backends. The TensorFlow.js demo uses TensorFlow.js 4.22.0 from a CDN, so its initial load requires network access. Browser/GPU support and available memory determine whether inference can run.

## Export model weights

The recommended source is Google's gated [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) checkpoint on Hugging Face. Accept the upstream terms and authenticate before exporting. Requires Python 3, `torch`, `numpy`, `safetensors`, and `huggingface_hub`.

```bash
python3 -m pip install torch numpy safetensors huggingface_hub transformers
hf auth login
python3 python/export_weights.py --source hf --output ./model
python3 python/export_chat_tokenizer.py
python3 python/prepare_model_release.py ./model
```

The exporter requires a **new or empty output directory**. The tokenizer export step obtains Google tokenizer data locally and generates verification probes; the preparation step copies that generated BPE tokenizer from `fixtures/official-model/chat-tokenizer.json`. To reproduce an immutable source snapshot, supply `--revision <HF_COMMIT_SHA>` when exporting. The optional Google Kaggle/PyTorch checkpoint path is supported with `--source kaggle --checkpoint /path/to/model.ckpt`. Do not mix export routes in one directory.

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
- The production model was subsequently loaded locally by the project maintainer. A separate independent Google-oracle comparison on the exact distribution package has not been recorded. The published Hugging Face FP32 package was loaded and used for generation on Mac.

The 512-token local-attention boundary was exercised, but full-vocabulary logit equivalence, cache ON/OFF equivalence beyond that boundary, all browser/driver combinations, and long-session stability are not claimed.

## Repository layout

- `src/backends/webgpu-turbo/`: optimized native WebGPU implementation
- `src/backends/webgpu/`: native WebGPU reference implementation
- `src/backends/tfjs-webgl/`: TensorFlow.js/WebGL implementation
- `src/tokenizer/`: Gemma 3 BPE tokenizer, chat template and detokenizer
- `app/`: static chat demos and validation utilities
- `python/`: model export and reference-comparison utilities
- `fixtures/` and `tests/`: test data and regression tests

## License and upstream terms

The **original source code** in this repository is licensed under [0BSD](LICENSE). This license does **not** apply to Google's Gemma 3 model weights or tokenizer assets. Any use or redistribution of those model materials must comply with the [Gemma Terms of Use](https://gemma.ai/resources/terms/) and [Gemma Prohibited Use Policy](https://gemma.ai/resources/prohibited-use/). Converted model distributions must include the applicable agreement copy and the required `Notice` and modification disclosures. The source-only repository excludes the Google-derived tokenizer fixture; regenerate it locally with `python/export_chat_tokenizer.py` or use an existing model directory that already contains it. Model redistribution requires separate compliance, including the Section 3.2 use restrictions; see the [official agreement](https://ai.google.dev/gemma/terms). Conversion scripts do not automatically package these legal documents.

Gemma is a Google model; this independent project is not affiliated with or endorsed by Google.

## Performance (W16A32)

Reference FP32 remains the default for existing model exports and Hugging Face.
WebGL remains FP32 only. WebGPU detects the precision from the manifest.
Performance stores projection and tied embedding/head matrices in binary16;
normalization weights, activations, accumulation and KV cache remain FP32.
The device must support `shader-f16`. No automatic precision fallback is used.

Convert your existing full FP32 model directory (requires NumPy):

```sh
python3 python/convert_performance.py --input model/REFERENCE_DIRECTORY --output model/performance
python3 -m http.server 8000
```

Open `http://localhost:8000/app/webgpu/` and select Bundled model (Performance),
or select Local folder and choose `model/performance`. The converted directory
contains independent LM-head parts, so bundled loading does not require HTTP
Range support. Keep all parts and the tokenizer together. The HF option still
loads the original FP32 repository; no FP16 upload is required.

For a first comparison, use the same prompt and token limit in both modes,
record prefill, TTFT, decode and generated token IDs. The existing official
parity checks are intended for Reference; reduced precision may differ.
Mac browser generation was verified; reduced-precision numerical equivalence is not claimed. The bundled option uses HTTP resources and does not itself implement
an offline web cache or an iOS application package.

## Mobile (W4A32)

Convert the original full FP32 export (not the Performance export):

```sh
python3 python/convert_mobile.py --input model/REFERENCE_DIRECTORY --output model/mobile
```

Select Bundled model (Mobile INT4) in the WebGPU chat for a local server, or Hugging Face (Mobile INT4) to fetch the published model from `NZM0/gemma3-js-1b-it-int4`. The latter requires that the model repository has been uploaded and made public. It downloads individual packed tensors and head chunks without HTTP range requests. Remote Mobile loading with Reference was exercised on iPhone; Turbo was tested with bundled weights over local HTTPS. Each group of 32
weights uses a float32 scale and four packed uint32 words, 5 bits per weight
including scale overhead. The symmetric quantizer uses values -7 through 7,
encoded with offset 8. Matrix input dimensions must be multiples of 32.
Embedding rows are decoded from the same resident head buffers. KV cache,
normalization and accumulation remain FP32; full context capacity is unchanged.
Browser generation was verified on Mac M4 and iPhone 16e. Long-session stability, thermal effects and broad generation quality remain unverified.
Execution logs are mirrored to console and best-effort localStorage; GPU
allocation counts exclude driver allocations. Remote server log forwarding
is not included. Nonresident diagnostic logits are not supported in INT4 mode.

## Historical Reference performance

Maintainer measurements using WebGPU, the prompt `Where is the capital of Japan?`,
16 prompt tokens, KV caching enabled and 24 generated tokens. Single runs, not
controlled benchmark averages; generated sequences may differ by precision.

| Device | Precision | Prefill (ms) | TTFT (ms) | Decode (tokens/s) |
| --- | --- | ---: | ---: | ---: |
| Mac M4 | FP32 | 543.8 | 575.0 | 9.84 |
| Mac M4 | W16A32 | 361.7 | 384.5 | 12.09 |
| Mac M4 | W4A32 | 558.9 | 576.1 | 10.11 |
| iPhone 16e | W4A32 | 1644.0 | 1692.0 | 3.88 |

Tracked GPU buffer allocations at model load completion on Mac: FP32
4,294,554,112 bytes (4095.6 MiB); W4A32 920,392,192 bytes (877.8 MiB),
a 78.6% reduction. These are tracked allocations, not total process memory.

On iPhone, the Mac served static model and application files over local HTTPS;
all inference ran in Safari on the phone. Use a trusted HTTPS certificate for
mobile access; localhost HTTP is suitable for desktop testing. No inference
server is required. FP32 and FP16 mobile loading did not complete in the
maintainer's tests; the exact cause was not established.
