/** Final release audit: one >512-token model run, trace, then unload. */
export function createGpuBufferAudit(device) {
    const devicePrototype = Object.getPrototypeOf(device);
    const probe = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
    const bufferPrototype = Object.getPrototypeOf(probe);
    probe.destroy();
    // The probe buffer is destroyed before instrumentation begins.
    // See the note below about device-level and browser-internal allocations.
    const originalCreate = devicePrototype.createBuffer;
    const originalDestroy = bufferPrototype.destroy;
    const live = new Map();
    let allocated = 0;
    let destroyed = 0;
    let created = 0;
    let removed = 0;
    let installed = false;
    try {
        devicePrototype.createBuffer = function (descriptor) {
            const buffer = originalCreate.call(this, descriptor);
            if (this === device) {
                const size = Number(descriptor.size);
                live.set(buffer, size);
                allocated += size;
                created += 1;
            }
            return buffer;
        };
        bufferPrototype.destroy = function () {
            if (live.has(this)) {
                destroyed += live.get(this);
                removed += 1;
                live.delete(this);
            }
            return originalDestroy.call(this);
        };
        installed = devicePrototype.createBuffer !== originalCreate &&
            bufferPrototype.destroy !== originalDestroy;
    } catch (error) {
        devicePrototype.createBuffer = originalCreate;
        bufferPrototype.destroy = originalDestroy;
        throw new Error(`GPUBuffer instrumentation unsupported: ${error.message}`);
    }
    if (!installed) {
        devicePrototype.createBuffer = originalCreate;
        bufferPrototype.destroy = originalDestroy;
        throw new Error('GPUBuffer instrumentation cannot patch browser prototypes');
    }
    return {
        snapshot() {
            return { created, destroyed: removed, allocatedBytes: allocated,
                destroyedBytes: destroyed, liveCount: live.size,
                liveBytes: allocated - destroyed };
        },
        stop() {
            devicePrototype.createBuffer = originalCreate;
            bufferPrototype.destroy = originalDestroy;
        }
    };
}

export async function runFinalAudit({ engine, tokenizer, backend, tf, unload, beforeLoad, gpuAudit, log }) {
    const report = {
        schema: 'gemma3-js-final-release-audit-v1', backend,
        timestamp: new Date().toISOString(),
        beforeLoad: beforeLoad ?? null,
        beforeUnload: null, afterUnload: null,
        gpuBeforeUnload: null, gpuAfterUnload: null,
        trace: null, checks: [], status: 'running'
    };
    const snapshot = () => tf ? {
        numTensors: tf.memory().numTensors,
        numBytes: tf.memory().numBytes,
        numDataBuffers: tf.memory().numDataBuffers,
        unreliable: tf.memory().unreliable
    } : null;
    const check = (name, passed, detail) => {
        report.checks.push({ name, passed, detail });
        log(`${passed ? 'PASS' : 'FAIL'} ${name}: ${JSON.stringify(detail)}`);
    };
    try {
        const count = content => tokenizer.encodeMessages([{ role: 'user', content }]).length;
        let repetitions = 1;
        while (count(' 1'.repeat(repetitions)) < 520 && repetitions < 4096) {
            repetitions *= 2;
        }
        if (count(' 1'.repeat(repetitions)) < 520) {
            throw new Error('Could not build 520-token prompt');
        }
        let lo = Math.floor(repetitions / 2);
        let hi = repetitions;
        while (lo + 1 < hi) {
            const middle = Math.floor((lo + hi) / 2);
            if (count(' 1'.repeat(middle)) < 520) { lo = middle; }
            else { hi = middle; }
        }
        const prompt = ' 1'.repeat(hi);
        const promptTokenIds = tokenizer.encodeMessages([{ role: 'user', content: prompt }]);
        if (promptTokenIds.length > engine.maxContextTokens - 16) {
            throw new Error('Long prompt exceeds model context allowance');
        }
        engine.reset();
        engine.setUseCache(true);
        const generatedTokenIds = [];
        const selectedLogits = [];
        let prefill = null;
        await engine.chat(prompt, {
            maxNewTokens: 3,
            onPrefill: value => { prefill = value; },
            onToken: value => {
                generatedTokenIds.push(value.tokenId);
                selectedLogits.push(value.logit);
            }
        });
        report.trace = {
            schema: 'gemma3-js-parity-trace-v1',
            model: 'google/gemma-3-1b-it', backend,
            turns: [{ promptTokenIds, generatedTokenIds, selectedLogits }]
        };
        check('boundary-trace-generated', promptTokenIds.length > 512 &&
            generatedTokenIds.length > 0 && generatedTokenIds.length === selectedLogits.length,
            { promptTokens: promptTokenIds.length, generatedTokens: generatedTokenIds.length, prefill });
    } catch (error) {
        report.error = String(error.stack || error);
        log(`Final audit inference error: ${report.error}`);
    } finally {
        // Unload even when generation fails; this audit intentionally ends the session.
        try {
            engine.reset();
            report.beforeUnload = snapshot();
            report.gpuBeforeUnload = gpuAudit?.snapshot() ?? null;
            unload();
            if (backend === 'native-webgpu') {
                // destroy() is synchronous; the GPU may release physical storage later.
                report.gpuAfterUnload = gpuAudit?.snapshot() ?? null;
                if (report.gpuAfterUnload) {
                    check('webgpu-buffer-destroy-balance',
                        report.gpuAfterUnload.liveCount === 0 && report.gpuAfterUnload.liveBytes === 0,
                        report.gpuAfterUnload);
                } else {
                    check('webgpu-buffer-destroy-balance', false, 'GPUBuffer tracker not available');
                }
            } else {
                await tf.nextFrame();
                report.afterUnload = snapshot();
                const baseline = report.beforeLoad;
                check('tfjs-unload-returns-to-baseline', !!baseline &&
                    report.afterUnload.numTensors === baseline.numTensors &&
                    report.afterUnload.numBytes === baseline.numBytes &&
                    report.afterUnload.numDataBuffers === baseline.numDataBuffers,
                    { baseline, afterUnload: report.afterUnload });
            }
        } catch (error) {
            report.unloadError = String(error.stack || error);
            check('unload-completed', false, report.unloadError);
        }
        report.status = report.error || report.unloadError ? 'error' :
            report.checks.every(item => item.passed) ? 'passed' : 'failed';
    }
    return report;
}

export function downloadFinalAudit(report) {
    const blob = new Blob([JSON.stringify(report, null, 4)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `gemma3-${report.backend}-final-audit.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
