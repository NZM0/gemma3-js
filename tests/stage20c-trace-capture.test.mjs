import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../app/webgpu/chat.js', import.meta.url), 'utf8');

assert.match(source, /useCache: engine\.useCache/);
assert.match(source, /executionPath: engine\.useCache \? 'cache-on' : 'full-prefix-recompute'/);
assert.match(source, /activeTurn\.status = 'completed'/);
assert.match(source, /activeTurn\.status = 'failed'/);
assert.match(source, /activeTurn\.error = String\(error\.stack \|\| error\.message\)/);
assert.match(source, /if \(activeTurn\) \{\s*turns\.push/);
assert.match(source, /if \(turns\.length === 0\)/);

console.log('Stage 20C trace capture source checks PASS');
