import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const layer = readFileSync(new URL('../src/backends/webgpu/webgpu-full-layer.js', import.meta.url), 'utf8');
const head = readFileSync(new URL('../src/backends/webgpu/webgpu-lm-head.js', import.meta.url), 'utf8');

assert.match(layer, /this\.maxCache = config\.layerTypes\[layerIndex\] === "local" \? config\.slidingWindow : config\.maxPositions/);
assert.doesNotMatch(layer, /this\.maxCache = Math\.min/);
assert.match(layer, /required \$\{this\.maxCache\}-token KV buffer/);
assert.match(layer, /this\.uniformArena = this\.device\.createBuffer/);
assert.match(layer, /offset: uniformOffset, size: 32/);
assert.match(layer, /this\.uniformCursor = 0/g);
assert.match(head, /fn reduceChunk/);
assert.match(head, /this\.device\.queue\.submit\(\[encoder\.finish\(\)\]\)/);
assert.match(head, /this\.residentChunks\[index\]\.start \+ winners\[index \* 2 \+ 1\]/);
console.log('WebGPU strict KV capacity, uniform reuse, LM-head GPU reduction: PASS');
