import { decoderStep } from "./gemma3-decoder.js";
import { SessionCache } from "./gemma3-core.js";

/** Reference cache retains all past KV entries. Attention masking, rather than
 * cache eviction, decides which positions are visible. */
export class UnboundedSessionCache extends SessionCache {
    append(layerIndex, position, keys, values) {
        const layer = this.layers[layerIndex];
        layer.positions.push(position);
        layer.keys.push(keys);
        layer.values.push(values);
    }
}

function errorMetrics(a, b) {
    if (a.length !== b.length) {
        throw new Error("Mismatched vector lengths");
    }

    let maximum = 0;
    let squares = 0;

    for (let i = 0; i < a.length; i += 1) {
        const difference = Math.abs(a[i] - b[i]);

        if (!Number.isFinite(difference)) {
            throw new Error(`Non-finite difference at ${i}`);
        }

        maximum = Math.max(maximum, difference);
        squares += difference * difference;
    }

    return {
        maxAbsoluteError: maximum,
        rmsError: Math.sqrt(squares / a.length)
    };
}

/** Compare eviction-based KV cache against full-history masked attention.
 * Both use decoderStep's math; this specifically isolates cache eviction. */
export function compareWindowCaches(layer, inputs, config, layerIndex = 0, tolerance = 0) {
    if (!Number.isInteger(config.slidingWindow) || config.slidingWindow < 1) {
        throw new Error("slidingWindow must be a positive integer");
    }

    if (config.layerTypes[layerIndex] !== "local") {
        throw new Error("compareWindowCaches requires a local-attention layer");
    }

    const bounded = new SessionCache(config.numLayers);
    const unbounded = new UnboundedSessionCache(config.numLayers);
    const checkpoints = [];
    let maximum = 0;

    for (let position = 0; position < inputs.length; position += 1) {
        const actual = decoderStep(inputs[position], layer, bounded, layerIndex, position, config);
        const expected = decoderStep(inputs[position], layer, unbounded, layerIndex, position, config);
        const error = errorMetrics(actual, expected);
        const firstVisible = Math.max(0, position - config.slidingWindow + 1);
        const expectedPositions = Array.from(
            { length: position - firstVisible + 1 },
            (_, index) => firstVisible + index
        );
        const positions = bounded.layers[layerIndex].positions;
        const cachePositionsMatch = positions.length === expectedPositions.length
            && positions.every((value, index) => value === expectedPositions[index]);
        const matches = error.maxAbsoluteError <= tolerance && cachePositionsMatch;

        maximum = Math.max(maximum, error.maxAbsoluteError);

        if (position === 0 || position === config.slidingWindow - 1
            || position === config.slidingWindow || position === config.slidingWindow + 1
            || position === inputs.length - 1 || !matches) {
            checkpoints.push({
                position,
                cacheFirstPosition: positions[0],
                cacheLastPosition: positions[positions.length - 1],
                cacheLength: positions.length,
                error,
                matches
            });
        }

        if (!matches) {
            return { matches: false, maximumError: maximum, checkpoints };
        }
    }

    return { matches: true, maximumError: maximum, checkpoints };
}
