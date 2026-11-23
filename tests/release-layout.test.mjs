import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { Gemma3ConversationEngine, TfjsFullModelSession, WebGpuFullModelSession } from "../src/index.js";

assert.equal(typeof Gemma3ConversationEngine, "function");
assert.equal(typeof TfjsFullModelSession, "function");
assert.equal(typeof WebGpuFullModelSession, "function");
assert.ok(existsSync("app/index.html"));
assert.ok(existsSync("app/webgpu/index.html"));
assert.ok(existsSync("app/tfjs/index.html"));
assert.ok(readFileSync("app/webgpu/chat.js", "utf8").includes("../src/"));
console.log("Release layout: PASS");
