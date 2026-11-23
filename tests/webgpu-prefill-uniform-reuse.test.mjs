import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const parallel = readFileSync(new URL("../src/backends/webgpu/webgpu-parallel-prefill.js", import.meta.url), "utf8");
const model = readFileSync(new URL("../src/backends/webgpu/webgpu-full-model.js", import.meta.url), "utf8");

test("parallel prefill uses per-layer aligned uniform records instead of per-dispatch GPU buffers", () => {
    assert.match(parallel, /const uniform = layer\.uniformArena/);
    assert.match(parallel, /layer\.uniformCursor \+= 1/);
    assert.match(parallel, /offset: uniformOffset, size: 48/);
    assert.doesNotMatch(parallel, /device\.createBuffer\(\{ size: 48, usage: GPUBufferUsage\.UNIFORM/);
    assert.match(model, /layer\.prepareBatch\(1\)/);
});
