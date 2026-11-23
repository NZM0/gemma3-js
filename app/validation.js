/** Browser-side combined validation. Runs on the already loaded full model. */
export async function runCombinedValidation({ engine, tokenizer, session, backend, tf, log }) {
    const report = {
        schema: 'gemma3-js-combined-validation-v1',
        backend,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        checks: [],
        memory: [],
        status: 'running'
    };

    const snapshot = label => {
        const sample = { label, time: performance.now() };
        if (tf) {
            const m = tf.memory();
            sample.tfjs = {
                numTensors: m.numTensors,
                numBytes: m.numBytes,
                numDataBuffers: m.numDataBuffers,
                unreliable: m.unreliable,
                reasons: m.reasons
            };
        }
        report.memory.push(sample);
        return sample;
    };

    const record = (name, passed, details = {}) => {
        report.checks.push({ name, passed, ...details });
        log(`${passed ? 'PASS' : 'FAIL'} ${name}: ${JSON.stringify(details)}`);
    };

    const run = async (prompt, useCache, maxNewTokens = 1) => {
        engine.reset();
        engine.setUseCache(useCache);
        const generatedTokenIds = [];
        const selectedLogits = [];
        let prefill = null;
        const result = await engine.chat(prompt, {
            maxNewTokens,
            onPrefill: info => { prefill = info; },
            onToken: info => {
                generatedTokenIds.push(info.tokenId);
                selectedLogits.push(info.logit);
            }
        });
        return { generatedTokenIds, selectedLogits, prefill, response: result.text };
    };

    try {
        snapshot('loaded-idle');

        // Compare actual full-prefix recomputation with cache-on for a short prompt.
        const shortPrompt = '日本の首都はどこですか？';
        const on = await run(shortPrompt, true, 2);
        snapshot('short-cache-on');
        const off = await run(shortPrompt, false, 2);
        const errors = on.selectedLogits.map((value, i) => Math.abs(value - off.selectedLogits[i]));
        const error = Math.max(...errors);
        const tokenMatch = on.generatedTokenIds.length === off.generatedTokenIds.length &&
            on.generatedTokenIds.every((id, i) => id === off.generatedTokenIds[i]);
        record('cache-on-off-short', tokenMatch && errors.length >= 2 && error <= 0.001, {
            cacheOn: on,
            cacheOff: off,
            selectedLogitAbsoluteError: error,
            perTokenAbsoluteErrors: errors,
            comparedTokens: errors.length,
            noCacheImplementation: typeof session.forwardNoCache === 'function'
        });
        if (typeof session.forwardNoCache !== 'function') {
            report.checks[report.checks.length - 1].passed = false;
        }
        engine.reset();
        snapshot('after-short-reset');

        // Cross the 512-key boundary during prefill. Avoid relying on an exact
        // 507–511 token count: BPE merge lengths are not necessarily contiguous.
        const tokenCount = text => tokenizer.encodeMessages([{ role: 'user', content: text }]).length;
        const unit = ' 1';
        let lower = 1;
        let upper = 1;
        while (tokenCount(unit.repeat(upper)) < 520 && upper < 4096) {
            lower = upper;
            upper *= 2;
        }
        if (tokenCount(unit.repeat(upper)) < 520) {
            throw new Error('Unable to construct a prompt above 512 tokens');
        }
        while (lower + 1 < upper) {
            const middle = Math.floor((lower + upper) / 2);
            if (tokenCount(unit.repeat(middle)) >= 520) {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        const longPrompt = unit.repeat(upper);
        const actualCount = tokenCount(longPrompt);
        if (actualCount < 520 || actualCount > engine.maxContextTokens - 64) {
            throw new Error(`Long prompt size out of bounds: ${actualCount}`);
        }
        log(`Long-context input: ${actualCount} tokens (crosses 512 during prefill)`);

        engine.reset();
        engine.setUseCache(true);
        const firstPromptTokens = tokenCount(longPrompt);
        let firstPrefill = null;
        const first = await engine.chat(longPrompt, {
            maxNewTokens: 4,
            onPrefill: info => { firstPrefill = info; }
        });
        const positionAfterFirst = session.position;
        snapshot('after-long-first-turn');
        let secondPrefill = null;
        const second = await engine.chat('続けてください。', {
            maxNewTokens: 2,
            onPrefill: info => { secondPrefill = info; }
        });
        const positionAfterSecond = session.position;
        const crossed = positionAfterFirst > 512 || positionAfterSecond > 512;
        record('real-model-window-crossing', crossed && firstPrefill.total === firstPromptTokens, {
            firstPromptTokens,
            positionAfterFirst,
            positionAfterSecond,
            firstPrefill,
            secondPrefill,
            firstResponse: first.text,
            secondResponse: second.text
        });
        record('multi-turn-cache-consistency', secondPrefill.reused > 0 && secondPrefill.processed > 0, {
            secondPrefill,
            note: 'A zero reused count indicates a prefix mismatch and full prefill; investigate chat template and cache policy.'
        });
        snapshot('after-long-second-turn');
        engine.reset();
        snapshot('after-long-reset');

        // Warm-up is complete. Compare repeated post-reset counts, not
        // the pre-first-inference baseline (which may contain lazy allocations).
        const counts = [];
        const bytes = [];
        for (let i = 0; i < 3; i += 1) {
            await run(shortPrompt, true, 1);
            engine.reset();
            const sample = snapshot(`repeat-reset-${i + 1}`);
            if (sample.tfjs) {
                counts.push(sample.tfjs.numTensors);
                bytes.push(sample.tfjs.numBytes);
            }
        }
        if (tf) {
            const stable = counts.every(value => value === counts[0]) &&
                bytes.every(value => value === bytes[0]);
            record('tfjs-reset-tensor-stability', stable, { counts, bytes,
                initialLoadedIdle: report.memory[0].tfjs,
                note: 'Repeated-reset stability is necessary but not proof of full unload cleanup.' });
        } else {
            record('webgpu-reset-completed', true, {
                note: 'No standardized GPU memory counter is available; inspect browser GPU process after unload.'
            });
        }

        report.status = report.checks.every(check => check.passed) ? 'passed' : 'failed';
    } catch (error) {
        report.status = 'error';
        report.error = String(error.stack || error.message);
        log(`ERROR combined validation: ${report.error}`);
    } finally {
        engine.reset();
        snapshot('final-reset');
    }
    return report;
}

export function downloadValidationReport(report) {
    const blob = new Blob([JSON.stringify(report, null, 4)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `gemma3-${report.backend}-combined-validation.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
