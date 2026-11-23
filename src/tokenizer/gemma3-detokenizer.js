/** SentencePiece-style Gemma token decoder. Uses official ID-to-piece export. */
export class Gemma3Detokenizer {
    constructor(pieces, specialTokenIds = []) {
        if (!Array.isArray(pieces) || pieces.length === 0) {
            throw new Error("Expected a nonempty token-piece array");
        }
        this.pieces = pieces;
        this.specialTokenIds = new Set(specialTokenIds);

        // Some Gemma chat delimiters (notably <end_of_turn>) are not
        // necessarily included in tokenizer.all_special_ids.
        // Detect only known structural markers, never arbitrary <...> text.
        const chatMarkers = new Set([
            "<start_of_turn>",
            "<end_of_turn>",
            "<|im_start|>",
            "<|im_end|>",
        ]);

        for (let id = 0; id < pieces.length; id += 1) {
            if (chatMarkers.has(pieces[id])) {
                this.specialTokenIds.add(id);
            }
        }
    }

    decode(tokenIds, { skipSpecialTokens = true } = {}) {
        const bytes = [];
        const encoder = new TextEncoder();

        for (const id of tokenIds) {
            if (!Number.isInteger(id) || id < 0 || id >= this.pieces.length) {
                throw new Error(`Invalid token ID ${id}`);
            }
            if (skipSpecialTokens && this.specialTokenIds.has(id)) {
                continue;
            }

            const piece = this.pieces[id];
            if (typeof piece !== "string") {
                throw new Error(`Missing token piece for ID ${id}`);
            }

            const byteMatch = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
            if (byteMatch) {
                bytes.push(parseInt(byteMatch[1], 16));
            } else {
                const expanded = piece.replaceAll("▁", " ");
                bytes.push(...encoder.encode(expanded));
            }
        }

        // SentencePiece uses a dummy leading space, removed at sequence start.
        if (bytes[0] === 0x20) {
            bytes.shift();
        }
        return new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(bytes));
    }
}
