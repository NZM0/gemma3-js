import assert from "node:assert/strict";
import { kvWindowPlan } from "../src/gemma3-kv-window-plan.js";

function runWindow(capacity, total, local) {
    const ring = Array(capacity).fill(null);
    const checkpoints = [];

    for (let position = 0; position < total; position += 1) {
        const plan = kvWindowPlan(position, capacity);
        ring[plan.writeSlot] = position;

        const actual = Array.from({ length: plan.length }, (_, t) =>
            ring[(plan.firstSlot + t) % capacity]
        );
        const expected = Array.from({ length: plan.length }, (_, t) =>
            plan.firstPosition + t
        );

        assert.deepEqual(actual, expected, `KV order mismatch at position ${position}`);
        assert.equal(actual.at(-1), position, "Current key must be visible");

        if (local) {
            assert.ok(actual.every(key => position - key < capacity));
        } else {
            assert.equal(actual[0], 0, "Global cache must preserve first key");
        }

        if ([0, 510, 511, 512, 513, 514, 519].includes(position)) {
            checkpoints.push({ position, firstKey: actual[0], lastKey: actual.at(-1), count: actual.length });
        }
    }

    return checkpoints;
}

const local = runWindow(512, 520, true);
const global = runWindow(520, 520, false);
assert.deepEqual(local.map(x => x.firstKey), [0, 0, 0, 1, 2, 3, 8]);
assert.deepEqual(global.map(x => x.firstKey), [0, 0, 0, 0, 0, 0, 0]);
assert.throws(() => kvWindowPlan(-1, 512), RangeError);
assert.throws(() => kvWindowPlan(0, 0), RangeError);

// Check the *real WGSL backend* uses this plan in both the model and diagnostic paths.
const { readFileSync } = await import("node:fs");
const source = readFileSync(new URL("../src/backends/webgpu/webgpu-full-layer.js", import.meta.url), "utf8");
assert.equal((source.match(/kvWindowPlan\(/g) ?? []).length, 2);
assert.ok(source.includes("const slot = plan.writeSlot"));
assert.ok(source.includes("let firstPosition = p.position + 1u - keys;"));
assert.ok(source.includes("% capacity) * keyStride"));
assert.ok(!source.includes("const ordered = this.make"));
assert.ok(source.includes("const length = plan.length"));
assert.ok(source.includes("config.slidingWindow : config.maxPositions"), "Local KV capacity must equal sliding window size (512)");

console.log("Stage 20B: WebGPU KV window ordering tests PASS");
console.log(JSON.stringify({ local, global }, null, 4));
