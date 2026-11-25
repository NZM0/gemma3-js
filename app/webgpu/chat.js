import { createLocalModelSource, createRemoteModelSource } from '../../src/model-source.js';
import { runFinalAudit, downloadFinalAudit, createGpuBufferAudit } from '../final-audit.js';
import { runCombinedValidation, downloadValidationReport } from '../validation.js';
import { GEMMA3_1B } from '../../src/gemma3-core.js';
import { Gemma3ChatTokenizer } from '../../src/tokenizer/gemma3-chat-tokenizer.js';
import { WebGpuFullLayerSession } from '../../src/backends/webgpu/webgpu-full-layer.js';
import { WebGpuFullModelSession } from '../../src/backends/webgpu/webgpu-full-model.js';
import { WebGpuLmHead } from '../../src/backends/webgpu/webgpu-lm-head.js';
import { Gemma3ConversationEngine } from '../../src/gemma3-conversation-engine.js';

const el = id => document.getElementById(id);
const folder = el('folder');
const loadButton = el('load');
const unloadButton = el('unload');
const sendButton = el('send');
const log = message => { el('log').textContent += `${message}\n`; };
const status = message => { el('status').textContent = message; log(message); };
let runtime = null;
let busy = false;
let engine = null;
let parityTrace = [];
let activeTurn = null;
let auditBeforeLoad = null;
let gpuBufferAudit = null;

const source = el('model-source');
const canLoad = () => source.value === 'hf' || folder.files.length > 0;
function updateSource() {
    folder.disabled = busy || source.value === 'hf';
    source.disabled = busy;
    loadButton.disabled = busy || !canLoad();
}
folder.addEventListener('change', updateSource);
source.addEventListener('change', updateSource);
updateSource();

function message(role, content = '') {
    const node = document.createElement('div');
    node.className = `message ${role}`;
    const title = document.createElement('small');
    title.textContent = role === 'user' ? 'You' : 'Gemma 3';
    const body = document.createElement('span');
    body.textContent = content;
    node.append(title, body);
    el('messages').append(node);
    el('messages').scrollTop = el('messages').scrollHeight;
    return body;
}

function dispose() {
    runtime?.head.dispose();
    runtime?.session.dispose();
    runtime = null;
    engine = null;
    parityTrace = [];
    activeTurn = null;
    unloadButton.disabled = true;
    sendButton.disabled = true;
    el('validate-combined').disabled = true;
    el('validate-final').disabled = true;
    el('prompt').disabled = true;
}

unloadButton.addEventListener('click', () => { if (!busy) { dispose(); status('Model unloaded.'); } });

loadButton.addEventListener('click', async () => {
    if (busy) { return; }
    busy = true;
    updateSource();
    dispose();
    auditBeforeLoad = null;
    const layers = [];
    let session = null;
    let head = null;
    try {
        const sourceMode = source.value;
        const get = sourceMode === 'hf' ? createRemoteModelSource() : createLocalModelSource(folder.files);
        const read = async name => new Float32Array(await get(name).arrayBuffer());
        const manifest = JSON.parse(await get('manifest.json').text());
        if (manifest.head.mode !== 'full') { throw new Error('A full-vocabulary model export is required'); }
        const tokenizerData = JSON.parse(await get('chat-tokenizer.json').text());
        const tokenizer = new Gemma3ChatTokenizer(tokenizerData);
        let tokenizerVerified = false;
        try {
            tokenizer.verifyPreTokenizationProbes(tokenizerData.preTokenizationProbes);
            tokenizer.verifyProbes(tokenizerData.probes || []);
            if (!tokenizerData.probes || tokenizerData.probes.length < 8) {
                throw new Error('Tokenizer probes missing: re-export with Stage 17 script');
            }
            tokenizerVerified = true;
            status('Tokenizer reference checks PASS. Loading WebGPU model...');
        } catch (error) {
            status(`Tokenizer verification failed: ${error.message}. Reference prompt only.`);
        }
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) { throw new Error('WebGPU unavailable'); }
        status(`WebGPU shader-f16: ${adapter.features.has('shader-f16')}`);
        for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension', 'maxStorageBuffersPerShaderStage']) {
            status(`Adapter ${name}: ${adapter.limits[name]}`);
        }
        const device = await adapter.requestDevice();
        status(`Device maxBufferSize: ${device.limits.maxBufferSize}; maxStorageBufferBindingSize: ${device.limits.maxStorageBufferBindingSize}`);
        gpuBufferAudit?.stop();
        gpuBufferAudit = null;
        try {
            gpuBufferAudit = createGpuBufferAudit(device);
        } catch (error) {
            status(`GPU buffer instrumentation unavailable: ${error.message}`);
        }
        device.lost.then(info => { status(`GPU device lost: ${info.message}`); dispose(); });
        device.addEventListener('uncapturederror', event => status(`GPU validation error: ${event.error.message}`));
        const norm = await read(manifest.finalNorm);
        for (let i = 0; i < manifest.layers.length; i += 1) {
            const spec = manifest.layers[i];
            const tensors = {};
            for (const [name, entry] of Object.entries(spec.tensors)) {
                tensors[name] = await read(`${spec.directory}/${entry.file}`);
            }
            layers.push(new WebGpuFullLayerSession(device, tensors, GEMMA3_1B, i));
            await device.queue.onSubmittedWorkDone();
            status(`GPU layers ${i + 1}/${manifest.layers.length}`);
        }
        session = new WebGpuFullModelSession(device, layers, norm);
        head = new WebGpuLmHead(device, norm);
        status('Uploading LM-head weights to persistent GPU buffers...');
        await head.loadResidentWeights(get(manifest.head.file), manifest.head.shape[0], (done, total) => {
            if (done === total || done % (8192 * 8) === 0) {
                status(`LM-head weights resident: ${done}/${total} rows`);
            }
        });
        session.enableGpuOutput();
        runtime = { session, head, tokenizer, tokenizerVerified, referenceText: tokenizerData.referenceText, referenceIds: tokenizerData.referenceTokenIds, headFile: get(manifest.head.file), vocab: manifest.head.shape[0], device };
        unloadButton.disabled = false;
        sendButton.disabled = false;
        el('validate-combined').disabled = false;
        el('validate-final').disabled = false;
        el('prompt').disabled = false;
        engine = new Gemma3ConversationEngine({
            session,
            tokenizer,
            embed: id => embedding(id, runtime.headFile, sourceMode === 'hf' ? head : null),
            predict: hidden => predict(hidden, runtime),
            eosTokenIds: GEMMA3_1B.eosTokenIds,
            useCache: el('use-cache').checked
        });
        el('messages').replaceChildren();
        status('Ready. Enter a prompt.');
    } catch (error) {
        status(`ERROR: ${error.stack || error.message}`);
        head?.dispose();
        if (session) { session.dispose(); }
        else { for (const layer of layers) { layer.dispose(); } }
    } finally {
        busy = false;
        updateSource();
    }
});

async function predict(hidden, state) {
    return hidden instanceof Float32Array
        ? state.head.predictResident(hidden)
        : state.head.predictResidentBuffer(hidden);
}

async function embedding(tokenId, headFile, residentHead = null) {
    const rowBytes = GEMMA3_1B.hiddenSize * 4;
    const vector = residentHead ? await residentHead.readEmbedding(tokenId) : new Float32Array(await headFile.slice(tokenId * rowBytes, (tokenId + 1) * rowBytes).arrayBuffer());
    const scale = Math.sqrt(GEMMA3_1B.hiddenSize);
    for (let i = 0; i < vector.length; i += 1) { vector[i] *= scale; }
    return vector;
}

el('new-chat').addEventListener('click', () => {
    if (busy) { return; }
    engine?.reset();
    el('messages').replaceChildren();
    parityTrace = [];
    status('New conversation started.');
});

el('chat').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !runtime) { return; }
    const prompt = el('prompt').value.trim();
    const maxTokens = Number(el('count').value);
    if (!prompt || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 64) { return; }
    busy = true;
    sendButton.disabled = true;
    unloadButton.disabled = true;
    message('user', prompt);
    const output = message('assistant');
    try {
        const state = runtime;
        if (!state.tokenizerVerified) {
            throw new Error('Tokenizer reference verification is required for conversation mode.');
        }
        if (engine.useCache !== el('use-cache').checked) {
            engine.setUseCache(el('use-cache').checked);
        }
        // Preserve K/V across turns when the setting has not changed.
        activeTurn = {
            prompt,
            maxNewTokens: maxTokens,
            useCache: engine.useCache,
            executionPath: engine.useCache ? 'cache-on' : 'full-prefix-recompute',
            promptTokenIds: [],
            generatedTokenIds: [],
            selectedLogits: [],
            status: 'running'
        };
        log(`Trace: useCache=${activeTurn.useCache}, executionPath=${activeTurn.executionPath}`);
        const result = await engine.chat(prompt, {
            maxNewTokens: maxTokens,
            onPrefill: ({ total, reused, processed }) => {
                activeTurn.promptTokenIds = engine.tokenizer.encodeMessages([...engine.messages, { role: 'user', content: prompt }]);
                activeTurn.prefill = { total, reused, processed };
                status(`Prefill: ${total} tokens (${reused} reused, ${processed} computed)`);
            },
            onToken: ({ tokenId, logit, text }) => {
                if (text !== null) { output.textContent = text; }
                activeTurn.generatedTokenIds.push(tokenId);
                activeTurn.selectedLogits.push(logit);
                log(`Generated: ${tokenId} (logit ${logit})`);
                el('messages').scrollTop = el('messages').scrollHeight;
            }
        });
        activeTurn.response = result.text;
        activeTurn.timing = result.timing;
        activeTurn.status = 'completed';
        parityTrace.push(activeTurn);
        activeTurn = null;
        const t = result.timing;
        log(`Timing: prefill=${t.prefillMs.toFixed(1)} ms, LM head=${t.headMs.toFixed(1)} ms, generation GPU steps=${t.generatedStepMs.toFixed(1)} ms`);
        log(`Timing: TTFT=${t.ttftMs.toFixed(1)} ms, decode=${t.decodeTokensPerSecond.toFixed(2)} tokens/sec`);
        el('performance').textContent = `Prefill ${t.prefillMs.toFixed(1)} ms | TTFT ${t.ttftMs.toFixed(1)} ms | Decode ${t.decodeTokensPerSecond.toFixed(2)} tok/s | LM head ${t.headMs.toFixed(1)} ms`;
        log(`Timing: LM head average=${(t.headMs / t.tokens).toFixed(1)} ms/token`);
        status(`Generation complete: ${result.generatedTokenIds.length} tokens`);
    } catch (error) {
        // Keep failed/partial turns for diagnosis instead of exporting turns: [].
        if (activeTurn) {
            activeTurn.status = 'failed';
            activeTurn.error = String(error.stack || error.message);
            activeTurn.partialResponse = output.textContent;
            parityTrace.push(activeTurn);
            activeTurn = null;
        }
        status(`ERROR: ${error.stack || error.message}`);
        output.textContent = `Generation error: ${error.message}`;
    } finally {
        busy = false;
        sendButton.disabled = runtime === null;
        unloadButton.disabled = runtime === null;
    }
});

// Stage 19D: capture reproducible per-turn inputs and selected-token logits.
// This trace is NOT a Google-reference result; it is the WebGPU side of comparison.
el('export-parity').addEventListener('click', () => {
    const turns = parityTrace.slice();
    if (activeTurn) {
        turns.push({
            ...activeTurn,
            status: 'running',
            note: 'Exported while generation was still in progress'
        });
    }
    const payload = {
        schema: 'gemma3-js-parity-trace-v1',
        model: 'google/gemma-3-1b-it',
        backend: 'native-webgpu',
        turns
    };
    if (turns.length === 0) {
        status('Trace is empty: submit a prompt and wait for completion (or an error) before exporting.');
        return;
    }
    const blob = new Blob([JSON.stringify(payload, null, 4)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gemma3-webgpu-trace.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// A single real-model run checks cache parity, the 512-token boundary,
// multi-turn reuse, and reset stability without loading model weights again.
el('validate-combined').addEventListener('click', async () => {
    if (busy || !runtime || !engine) { return; }
    if (!runtime.tokenizerVerified) {
        status('Tokenizer reference checks must pass before validation.');
        return;
    }
    busy = true;
    el('validate-combined').disabled = true;
    sendButton.disabled = true;
    unloadButton.disabled = true;
    try {
        status('Combined validation running. This may take several minutes on WebGL...');
        const report = await runCombinedValidation({
            engine,
            tokenizer: runtime.tokenizer,
            session: runtime.session,
            backend: 'native-webgpu',
            tf: null,
            log
        });
        downloadValidationReport(report);
        status(`Combined validation: ${report.status}. Report downloaded. Conversation reset.`);
    } finally {
        busy = false;
        el('validate-combined').disabled = runtime === null;
        sendButton.disabled = runtime === null;
        unloadButton.disabled = runtime === null;
    }
});


// Destructive final audit: generate the boundary trace and then unload the model.
el('validate-final').addEventListener('click', async () => {
    if (busy || !runtime || !engine) { return; }
    if (!runtime.tokenizerVerified) {
        status('Tokenizer reference checks must pass before final audit.');
        return;
    }
    busy = true;
    el('validate-final').disabled = true;
    el('validate-combined').disabled = true;
    sendButton.disabled = true;
    unloadButton.disabled = true;
    try {
        status('Final audit: boundary trace + resource cleanup. Model will be unloaded.');
        const report = await runFinalAudit({
            engine,
            tokenizer: runtime.tokenizer,
            backend: 'native-webgpu',
            tf: null,
            beforeLoad: auditBeforeLoad,
            gpuAudit: gpuBufferAudit,
            unload: dispose,
            log
        });
        downloadFinalAudit(report);
        status(`Final audit: ${report.status}. JSON downloaded; model unloaded.`);
    } finally {
        busy = false;
        el('validate-final').disabled = runtime === null;
        el('validate-combined').disabled = runtime === null;
        sendButton.disabled = runtime === null;
        unloadButton.disabled = runtime === null;
    }
});
