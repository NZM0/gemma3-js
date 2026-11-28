# WGSL Turbo (v0.3.0)

Independent optimized backend for Gemma 3 1B. Reference and TF.js are retained unchanged. Use `app/webgpu-turbo/` with existing model files; no additional conversion or runtime dependency is needed.

## Implementation

Workgroup RMSNorm and QK norm/RoPE; tiled online-softmax attention (64-key tiles); 8-lane INT4 row reduction for decode and resident head; tiled prefill GEMM; shared layer pipelines and prefill scratch buffers. Global FP32 KV buffers start at 2048 tokens and grow as required up to the configured context limit. Local ring capacity remains 512. Lifecycle audit checks growth, replay and tracked buffer destruction.

## Recorded validation

- Mac INT4: 24-token generation completed; cache ON/OFF short test matched 2/2 tokens, selected-logit maximum error 3.815e-5.
- Mac: 520-token prefill and subsequent cache reuse passed; 524 tokens reused.
- Mac: all four global caches grew from 2048 to 4096 at position 2049; selected token matched replay, logit error 5.722e-6. All 12,687 tracked buffers created during the lifecycle audit were destroyed; live bytes and count were zero.
- iPhone Safari: 24/24 generated tokens matched Mac Turbo, selected-logit maximum difference 8.774e-5; 13.58 tok/s.

These are limited selected-token tests, not full-vocabulary or independent official-oracle parity for Turbo. Long-session stability, thermal behavior, broad generation quality, iPhone growth/unload and all precision/device combinations remain unverified. Tracked buffer destruction does not establish immediate physical memory release. The lifecycle audit unloads the model.

Further optimization (split-KV, projection fusion, bind group caching and speculative decoding) is deferred.
