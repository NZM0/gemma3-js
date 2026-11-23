import { Gemma3Detokenizer } from './gemma3-detokenizer.js';
import { Gemma3Bpe } from './gemma3-bpe.js';

/** Local SentencePiece Unigram tokenizer. Verify reference IDs before arbitrary prompts. */
export class Gemma3ChatTokenizer {
    constructor(data) {
        this.data = data;
        this.tokenizerType = data.tokenizerType || "Unigram";
        this.decoder = new Gemma3Detokenizer(data.pieces, data.specialTokenIds);
        this.bpe = this.tokenizerType === "BPE" ? new Gemma3Bpe(data) : null;
        this.byPiece = new Map();
        this.maxPieceLength = 0;
        for (let id = 0; id < data.pieces.length; id += 1) {
            const piece = data.pieces[id];
            if (piece.startsWith('<') && piece.endsWith('>')) { continue; }
            if (this.tokenizerType !== "Unigram" || !Number.isFinite(data.scores?.[id])) { continue; }
            this.byPiece.set(piece, { id, score: data.scores[id] });
            this.maxPieceLength = Math.max(this.maxPieceLength, [...piece].length);
        }
        this.special = new Map(data.pieces.map((piece, id) => [piece, id]));
    }

    encodeText(text) {
        if (this.bpe) { return this.bpe.encode(text); }
        // Gemma SentencePiece: prepend dummy prefix, replace spaces with U+2581.
        const normalized = `▁${text.replaceAll(' ', '▁')}`;
        const chars = [...normalized];
        const n = chars.length;
        const scores = new Float64Array(n + 1).fill(-Infinity);
        const previous = new Array(n + 1);
        scores[0] = 0;
        for (let i = 0; i < n; i += 1) {
            if (!Number.isFinite(scores[i])) { continue; }
            let piece = '';
            for (let j = i; j < Math.min(n, i + this.maxPieceLength); j += 1) {
                piece += chars[j];
                const entry = this.byPiece.get(piece);
                if (!entry) { continue; }
                const next = j + 1;
                const score = scores[i] + entry.score;
                if (score > scores[next]) {
                    scores[next] = score;
                    previous[next] = { start: i, ids: [entry.id] };
                }
            }
            // SentencePiece byte fallback for unseen UTF-8 characters.
            const bytes = new TextEncoder().encode(chars[i]);
            const ids = [...bytes].map(byte => this.special.get(`<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`));
            if (ids.every(id => id !== undefined) && scores[i] - 100 < scores[i + 1]) {
                scores[i + 1] = scores[i] - 100;
                previous[i + 1] = { start: i, ids };
            }
        }
        if (!previous[n]) { throw new Error('Tokenizer cannot encode the supplied text'); }
        const chunks = [];
        for (let i = n; i > 0;) {
            const item = previous[i];
            if (!item) { throw new Error(`Tokenizer path missing at ${i}`); }
            chunks.push(item.ids);
            i = item.start;
        }
        return chunks.reverse().flat();
    }

    encodeChat(text) {
        return this.encodeMessages([{ role: 'user', content: text }]);
    }

    encodeMessages(messages) {
        const bos = this.special.get('<bos>');
        const start = this.special.get('<start_of_turn>');
        const end = this.special.get('<end_of_turn>');

        if (bos === undefined || start === undefined || end === undefined) {
            throw new Error('Chat markers missing (bos/start_of_turn/end_of_turn)');
        }
        if (!Array.isArray(messages) || messages.length === 0) {
            throw new Error('At least one chat message is required');
        }

        const ids = [bos];

        for (let index = 0; index < messages.length; index += 1) {
            const item = messages[index];
            const expectedRole = index % 2 === 0 ? 'user' : 'assistant';

            if (item.role !== expectedRole) {
                throw new Error(`Expected ${expectedRole} at message ${index}, got ${item.role}`);
            }
            if (typeof item.content !== 'string') {
                throw new Error('Only text chat content is supported');
            }

            const role = item.role === 'assistant' ? 'model' : 'user';
            ids.push(start);
            ids.push(...this.encodeText(`${role}\n`));
            ids.push(...this.encodeText(item.content.trim()));
            ids.push(end);
            ids.push(...this.encodeText('\n'));
        }

        ids.push(start);
        ids.push(...this.encodeText('model\n'));
        return ids;
    }

    verifyPreTokenizationProbes(probes) {
        if (!this.bpe) { return true; }
        if (!Array.isArray(probes) || probes.length === 0) {
            throw new Error('BPE pre-tokenization reference probes missing');
        }
        for (const probe of probes) {
            const actual = this.bpe.preTokenize(probe.text);
            if (JSON.stringify(actual) !== JSON.stringify(probe.pieces)) {
                throw new Error(`Pre-tokenizer mismatch: ${probe.name}; expected ${JSON.stringify(probe.pieces)}, got ${JSON.stringify(actual)}`);
            }
        }
        return true;
    }

    verifyProbes(probes) {
        for (const probe of probes) {
            const actual = probe.messages
                ? this.encodeMessages(probe.messages)
                : this.encodeText(probe.text);
            if (JSON.stringify(actual) !== JSON.stringify(probe.ids)) {
                throw new Error(`Tokenizer mismatch: ${probe.name}; expected ${JSON.stringify(probe.ids)}, got ${JSON.stringify(actual)}`);
            }
        }
        return true;
    }

    decode(ids) {
        return this.decoder.decode(ids);
    }
}
