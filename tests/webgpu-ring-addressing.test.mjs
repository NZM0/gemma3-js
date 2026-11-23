import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { kvWindowPlan } from "../src/gemma3-kv-window-plan.js";

// The WGSL shader uses (firstPosition + t) % capacity for every K/V read.
// Check the shader's exact address formula against the reference window plan,
// including wraparound and the first two positions beyond the 512 boundary.
for (const capacity of [1, 2, 7, 512, 1024]) {
    for (let position = 0; position < capacity * 2 + 3; position += 1) {
        const plan = kvWindowPlan(position, capacity);
        const firstPosition = position + 1 - plan.length;
        for (let t = 0; t < plan.length; t += 1) {
            const shaderSlot = (firstPosition + t) % capacity;
            const expectedSlot = (plan.firstSlot + t) % capacity;
            assert.equal(shaderSlot, expectedSlot, `capacity=${capacity}, position=${position}, t=${t}`);
        }
    }
}

const source = readFileSync(new URL("../src/backends/webgpu/webgpu-full-layer.js", import.meta.url), "utf8");
assert.ok(source.includes("let firstPosition = p.position + 1u - keys;"));
assert.ok(source.includes("((firstPosition + t) % capacity) * keyStride + j"));
assert.ok(source.includes("((firstPosition + t) % capacity) * keyStride + headDim + component"));
assert.ok(source.includes("view.setUint32(28, this.maxCache, true);"));
assert.ok(!source.includes("const ordered = this.make"));
console.log("WebGPU circular KV addressing: PASS");
