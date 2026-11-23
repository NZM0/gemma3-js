/** Tied embedding projection, partitioned to respect WebGL texture limits.
 * The mathematical operation is the same as the official full-vocabulary
 * projection. Only the physical tensor allocation and argmax reduction differ.
 */
export class TfjsOfficialHead {
    constructor(tf, weights, vocabSize, hiddenSize, chunkRows = 2048) {
        if (weights.length !== vocabSize * hiddenSize) {
            throw new Error("LM head weight shape mismatch");
        }
        if (!Number.isInteger(chunkRows) || chunkRows < 1) {
            throw new Error("chunkRows must be a positive integer");
        }

        this.tf = tf;
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.chunkRows = chunkRows;
        this.chunks = [];
        this.disposed = false;

        try {
            for (let start = 0; start < vocabSize; start += chunkRows) {
                const rows = Math.min(chunkRows, vocabSize - start);
                const offset = start * hiddenSize;
                const slice = weights.subarray(offset, offset + rows * hiddenSize);
                const tensor = tf.tensor2d(slice, [rows, hiddenSize]);
                this.chunks.push({ start, tensor });
            }
        } catch (error) {
            this.dispose();
            throw error;
        }
    }

    async predict(hidden) {
        if (this.disposed) {
            throw new Error("LM head has been disposed");
        }

        let bestId = -1;
        let bestLogit = -Infinity;

        try {
            const column = hidden.reshape([this.hiddenSize, 1]);
            try {
                for (const { start, tensor } of this.chunks) {
                    const result = this.tf.tidy(() => {
                        const logits = tensor.matMul(column).reshape([tensor.shape[0]]);
                        const index = logits.argMax();
                        const value = logits.gather(index.reshape([1])).reshape([]);
                        return { index, value };
                    });

                    try {
                        const [indexData, valueData] = await Promise.all([
                            result.index.data(),
                            result.value.data()
                        ]);
                        const candidateId = start + indexData[0];
                        const candidateLogit = valueData[0];
                        if (candidateLogit > bestLogit ||
                            (candidateLogit === bestLogit && candidateId < bestId)) {
                            bestId = candidateId;
                            bestLogit = candidateLogit;
                        }
                    } finally {
                        result.index.dispose();
                        result.value.dispose();
                    }
                }
            } finally {
                column.dispose();
            }
            return { bestId, bestLogit };
        } finally {
            hidden.dispose();
        }
    }

    dispose() {
        if (this.disposed) {
            return;
        }
        for (const { tensor } of this.chunks) {
            tensor.dispose();
        }
        this.chunks.length = 0;
        this.disposed = true;
    }
}
