import assert from "node:assert/strict";
import {
    GEMMA3_1B,
    rmsNorm,
    geluTanh,
    applyRopeHalfSplit,
    attentionForHead,
    visibleKeyPositions,
    SessionCache
} from "../src/gemma3-core.js";

function close(actual, expected, tolerance = 1e-5) {
    assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}

assert.equal(GEMMA3_1B.layerTypes[5], "global");
assert.equal(GEMMA3_1B.layerTypes[6], "local");
assert.equal(GEMMA3_1B.layerTypes[25], "local");

const normalized = rmsNorm([3, 4], [0, 0]);
close(normalized[0], 3 / Math.sqrt(12.5 + 1e-6));
close(normalized[1], 4 / Math.sqrt(12.5 + 1e-6));
close(geluTanh(0), 0);

const rotated = applyRopeHalfSplit([1, 0, 0, 1], 0);
assert.deepEqual(Array.from(rotated), [1, 0, 0, 1]);

assert.deepEqual(visibleKeyPositions(5, "local", 3), [3, 4, 5]);
assert.deepEqual(visibleKeyPositions(5, "global", 3), [0, 1, 2, 3, 4, 5]);

const result = attentionForHead(
    [1, 0],
    [[1, 0], [1, 0]],
    [[10, 0], [0, 20]],
    [0, 1],
    1,
    "global"
);
close(result[0], 5);
close(result[1], 10);

const cache = new SessionCache(1);
cache.append(0, 0, [1], [2], "local", 2);
cache.append(0, 1, [3], [4], "local", 2);
cache.append(0, 2, [5], [6], "local", 2);
assert.deepEqual(cache.layers[0].positions, [1, 2]);
cache.clear();
assert.equal(cache.layers[0].positions.length, 0);

console.log("Gemma 3 core: all tests passed");
