// One real-model growth test, followed by a destructive resource audit.
export async function runLifecycleAudit({ engine, session, gpuAudit, unload, log }) {
    const report = {
        schema: 'gemma3-js-turbo-lifecycle-v1',
        backend: 'wgsl-turbo',
        checks: [],
        status: 'running'
    };
    const check = (name, passed, detail) => {
        report.checks.push({ name, passed, detail });
        log(`${passed ? 'PASS' : 'FAIL'} ${name}: ${JSON.stringify(detail)}`);
    };
    const capacities = () => session.layers.map(layer => layer.maxCache);
    const predict = result => engine.predict(result.gpuHiddenBuffer ?? result.output);
    try {
        engine.setUseCache(true);
        engine.reset();
        const before = capacities();
        const global = session.layers.filter(layer => layer.config.layerTypes[layer.layerIndex] === 'global');
        if (!global.length || global.some(layer => layer.maxCache !== 2048)) {
            throw new Error('Load a fresh model before this audit (initial global capacity must be 2048).');
        }
        const seed = engine.tokenizer.encodeMessages([{ role: 'user', content: 'The forest is quiet. 森は静かです。' }]);
        const ids = Array.from({ length: 2049 }, (_, i) => seed[i % seed.length]);
        await engine.prefill(ids.slice(0, 2048));
        const beforeGrowth = gpuAudit?.snapshot() ?? null;
        const grown = await engine.stepToken(ids[2048]);
        const split = await predict(grown);
        const after = capacities();
        const afterGrowth = gpuAudit?.snapshot() ?? null;
        const position = session.position;
        check('dynamic-cache-growth', position === 2049 &&
            session.layers.every((layer, i) => layer.position === 2049 &&
                (layer.config.layerTypes[i] === 'global' ? after[i] === 4096 : after[i] === before[i])),
            { before, after, position, beforeGrowth, afterGrowth });
        engine.reset();
        const fresh = await engine.prefill(ids);
        const replay = await predict(fresh.last);
        const error = Math.abs(split.bestLogit - replay.bestLogit);
        check('growth-prefix-replay-parity', split.bestId === replay.bestId &&
            Number.isFinite(error) && error <= 0.01,
            { split, replay, selectedLogitAbsoluteError: error, tolerance: 0.01 });
    } catch (error) {
        report.error = String(error.stack || error);
        log(report.error);
    } finally {
        try {
            await session.device.queue.onSubmittedWorkDone();
            report.gpuBeforeUnload = gpuAudit?.snapshot() ?? null;
            unload();
            report.gpuAfterUnload = gpuAudit?.snapshot() ?? null;
            const after = report.gpuAfterUnload;
            check('webgpu-buffer-destroy-balance', !!after && after.liveCount === 0 && after.liveBytes === 0,
                after ?? 'GPUBuffer tracker unavailable');
        } catch (error) {
            report.unloadError = String(error.stack || error);
            check('unload-completed', false, report.unloadError);
        }
        report.status = report.error || report.unloadError ? 'error' :
            report.checks.every(item => item.passed) ? 'passed' : 'failed';
    }
    return report;
}
