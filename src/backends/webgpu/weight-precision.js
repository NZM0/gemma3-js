export function decodeHalfArray(values) {
    return Float32Array.from(values, bits => {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits >> 10) & 31;
        const fraction = bits & 1023;
        if (exponent === 31) { return fraction ? NaN : sign * Infinity; }
        if (exponent === 0) { return sign * fraction * 2 ** -24; }
        return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
    });
}

export function encodeHalfArray(values) {
    const roundEven = x => {
        const floor = Math.floor(x);
        return floor + Number(x - floor > 0.5 || (x - floor === 0.5 && floor % 2 === 1));
    };
    return Uint16Array.from(values, value => {
        const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
        const x = Math.abs(value);
        if (Number.isNaN(x)) { return 0x7e00; }
        if (x >= 65520) { return sign | 0x7c00; }
        if (x < 2 ** -14) { return sign | roundEven(x * 2 ** 24); }
        const exponent = Math.floor(Math.log2(x));
        return sign | (((exponent + 15) << 10) + roundEven((x / 2 ** exponent - 1) * 1024));
    });
}

export function decodeInt4(words) {
    if (words.length % 5) { throw new Error('Invalid INT4 group'); }
    const scales = new Float32Array(words.buffer, words.byteOffset, words.length);
    const result = new Float32Array(words.length / 5 * 32);
    for (let g = 0; g < words.length / 5; g += 1) {
        for (let j = 0; j < 32; j += 1) {
            const q = ((words[g * 5 + 1 + (j >> 3)] >>> ((j % 8) * 4)) & 15) - 8;
            result[g * 32 + j] = q * scales[g * 5];
        }
    }
    return result;
}
