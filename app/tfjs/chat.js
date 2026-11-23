import { runFinalAudit, downloadFinalAudit } from '../final-audit.js';
import { runCombinedValidation, downloadValidationReport } from '../validation.js';
import { GEMMA3_1B } from '../../src/gemma3-core.js';
import { Gemma3ChatTokenizer } from '../../src/tokenizer/gemma3-chat-tokenizer.js';
import { TfjsOfficialModel } from '../../src/backends/tfjs-webgl/tfjs-official-model.js';
import { TfjsOfficialHead } from '../../src/backends/tfjs-webgl/tfjs-official-head.js';
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

folder.addEventListener('change', () => { loadButton.disabled = folder.files.length === 0; });

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
    loadButton.disabled = true;
    dispose();
    auditBeforeLoad = window.tf?.memory() ? { numTensors: window.tf.memory().numTensors, numBytes: window.tf.memory().numBytes, numDataBuffers: window.tf.memory().numDataBuffers } : null;
    const layers = [];
    let session = null;
    let head = null;
    try {
        const files = new Map();
        for (const file of folder.files) {
            const parts = (file.webkitRelativePath || file.name).split('/');
            files.set(parts.slice(1).join('/'), file);
        }
        const get = name => {
            const file = files.get(name);
            if (!file) { throw new Error(`Missing file: ${name}`); }
            return file;
        };
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
            status('Tokenizer reference checks PASS. Loading TensorFlow.js WebGL model...');
        } catch (error) {
            status(`Tokenizer verification failed: ${error.message}. Reference prompt only.`);
        }
        const tf = window.tf;
        if (!tf) { throw new Error('TensorFlow.js did not load. Check the CDN connection.'); }
        await tf.setBackend('webgl');
        await tf.ready();
        if (tf.getBackend() !== 'webgl') { throw new Error('WebGL backend unavailable'); }
        const norm = await read(manifest.finalNorm);
        for (let i = 0; i < manifest.layers.length; i += 1) {
            const spec = manifest.layers[i];
            const tensors = {};
            for (const [name, entry] of Object.entries(spec.tensors)) {
                tensors[name] = await read(`${spec.directory}/${entry.file}`);
            }
            layers.push(tensors);
            status(`Model layer data ${i + 1}/${manifest.layers.length}`);
        }
        status('Creating TensorFlow.js layer weights...');
        session = new TfjsOfficialModel(tf, layers, norm);
        status('Loading tied LM head in WebGL-safe tensor partitions (approximately 1.13 GiB total)...');
        const headWeights = await read(manifest.head.file);
        head = new TfjsOfficialHead(tf, headWeights, manifest.head.shape[0], GEMMA3_1B.hiddenSize);
        runtime = { session, head, tokenizer, tokenizerVerified, referenceText: tokenizerData.referenceText, referenceIds: tokenizerData.referenceTokenIds, headFile: get(manifest.head.file), vocab: manifest.head.shape[0] };
        unloadButton.disabled = false;
        sendButton.disabled = false;
        el('validate-combined').disabled = false;
        el('validate-final').disabled = false;
        el('prompt').disabled = false;
        engine = new Gemma3ConversationEngine({
            session,
            tokenizer,
            embed: id => embedding(id, runtime.headFile),
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
        runtime = null;
        engine = null;
        
    } finally {
        busy = false;
        loadButton.disabled = folder.files.length === 0;
    }
});

async function predict(hidden, state) {
    return state.head.predict(hidden);
}

async function embedding(tokenId, headFile) {
    const rowBytes = GEMMA3_1B.hiddenSize * 4;
    const vector = new Float32Array(await headFile.slice(tokenId * rowBytes, (tokenId + 1) * rowBytes).arrayBuffer());
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
// This trace records TensorFlow.js/WebGL results, not a Google reference.
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
        backend: 'tfjs-webgl',
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
    a.download = 'gemma3-tfjs-webgl-trace.json';
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
            backend: 'tfjs-webgl',
            tf: window.tf,
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
            backend: 'tfjs-webgl',
            tf: window.tf,
            beforeLoad: auditBeforeLoad,
            gpuAudit: null,
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
