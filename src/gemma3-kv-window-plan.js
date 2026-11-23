/** Absolute-position window plan shared by both WebGPU layer execution paths.
 *  The write for `position` must occur before copying the ordered KV window.
 *  `firstPosition` is the absolute position of the first visible key.
 */
export function kvWindowPlan(position, capacity) {
    if (!Number.isSafeInteger(position) || position < 0) {
        throw new RangeError("position must be a non-negative safe integer");
    }
    if (!Number.isSafeInteger(capacity) || capacity < 1) {
        throw new RangeError("capacity must be a positive safe integer");
    }

    const length = Math.min(position + 1, capacity);
    const firstPosition = position + 1 - length;

    return {
        writeSlot: position % capacity,
        firstSlot: firstPosition % capacity,
        firstPosition,
        length
    };
}
