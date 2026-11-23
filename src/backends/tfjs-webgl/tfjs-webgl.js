/** TensorFlow.js WebGL matrix-vector projection adapter.
 * Pass a TensorFlow.js instance from the caller; no bundler required.
 * This is a tested-by-inspection backend primitive, NOT a complete Gemma 3 backend.
 */
export class TfjsWebglBackend {
    constructor(tf) {
        this.tf = tf;
    }

    async initialize() {
        await this.tf.setBackend("webgl");
        await this.tf.ready();

        if (this.tf.getBackend() !== "webgl") {
            throw new Error("TensorFlow.js WebGL backend unavailable");
        }
    }

    async linear(vector, rowMajorWeights, outputSize, inputSize) {
        if (vector.length !== inputSize || rowMajorWeights.length !== outputSize * inputSize) {
            throw new Error("TfjsWebglBackend.linear: dimension mismatch");
        }

        const tf = this.tf;
        const result = tf.tidy(() => {
            const input = tf.tensor2d(vector, [inputSize, 1], "float32");
            const weights = tf.tensor2d(rowMajorWeights, [outputSize, inputSize], "float32");
            return weights.matMul(input).reshape([outputSize]);
        });

        try {
            return Float32Array.from(await result.data());
        } finally {
            result.dispose();
        }
    }
}
