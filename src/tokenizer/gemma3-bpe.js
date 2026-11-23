/** Rank-based BPE for explicitly supported Hugging Face tokenizer configurations.
 * Unsupported tokenizer pipeline configurations fail closed, never silently tokenize.
 */
export class Gemma3Bpe {
    constructor(data) {
        const model = data.bpeModel;
        if (!model || model.type !== 'BPE') {
            throw new Error('Expected a BPE model');
        }
        this.vocab = new Map(Object.entries(model.vocab));
        this.unknown = model.unk_token;
        this.byteFallback = model.byte_fallback === true;
        this.continuingSubwordPrefix = model.continuing_subword_prefix || '';
        this.endOfWordSuffix = model.end_of_word_suffix || '';
        this.merges = new Map();
        for (let i = 0; i < model.merges.length; i += 1) {
            const merge = model.merges[i];
            const pair = Array.isArray(merge) ? merge : merge.split(' ');
            if (pair.length !== 2) { throw new Error('Unexpected BPE merge entry'); }
            this.merges.set(JSON.stringify(pair), i);
        }
        const pre = data.preTokenizer;
        this.stages = pre?.type === 'Sequence' ? pre.pretokenizers : [pre];
        if (!this.stages.length || this.stages.some(stage => !stage)) {
            throw new Error(`Unsupported BPE pre-tokenizer: ${JSON.stringify(pre)}`);
        }
        for (const stage of this.stages) {
            if (stage.type === 'Split') {
                if (stage.pattern?.String !== ' ' || stage.behavior !== 'MergedWithPrevious' || stage.invert !== false) {
                    throw new Error(`Unsupported BPE Split configuration: ${JSON.stringify(stage)}`);
                }
            } else if (stage.type === 'Metaspace') {
                const prepend = stage.prepend_scheme ?? (stage.add_prefix_space ? 'always' : 'never');
                if (!['always', 'never', 'first'].includes(prepend) || stage.split === false) {
                    throw new Error(`Unsupported Metaspace configuration: ${JSON.stringify(stage)}`);
                }
            } else {
                throw new Error(`Unsupported BPE pre-tokenizer: ${JSON.stringify(stage)}`);
            }
        }
        const normalizer = data.normalizer;
        this.normalizerStages = normalizer?.type === 'Sequence'
            ? normalizer.normalizers
            : normalizer == null ? [] : [normalizer];
        if (!Array.isArray(this.normalizerStages)) {
            throw new Error(`Unsupported BPE normalizer: ${JSON.stringify(normalizer)}`);
        }
        for (const stage of this.normalizerStages) {
            if (stage?.type !== 'Replace' || typeof stage.pattern?.String !== 'string' || typeof stage.content !== 'string' || stage.pattern.String.length === 0) {
                throw new Error(`Unsupported BPE normalizer: ${JSON.stringify(stage)}`);
            }
        }
        if (model.dropout !== null && model.dropout !== undefined) {
            throw new Error('BPE dropout is unsupported');
        }
    }

    normalize(text) {
        let result = text;
        for (const stage of this.normalizerStages) {
            // HF Replace with a String pattern performs literal global replacement.
            result = result.replaceAll(stage.pattern.String, stage.content);
        }
        return result;
    }

    preTokenize(text) {
        let pieces = [this.normalize(text)];
        for (const stage of this.stages) {
            if (stage.type === 'Split') {
                const next = [];
                for (const piece of pieces) {
                    // MergedWithPrevious: delimiter belongs to the preceding segment.
                    // Consecutive/leading spaces are retained as separate segments.
                    let start = 0;
                    for (let i = 0; i < piece.length; i += 1) {
                        if (piece[i] === ' ') {
                            next.push(piece.slice(start, i + 1));
                            start = i + 1;
                        }
                    }
                    if (start < piece.length) { next.push(piece.slice(start)); }
                }
                pieces = next.filter(piece => piece.length > 0);
            } else if (stage.type === 'Metaspace') {
                const replacement = stage.replacement ?? '▁';
                const prepend = stage.prepend_scheme ?? (stage.add_prefix_space ? 'always' : 'never');
                const next = [];
                for (let i = 0; i < pieces.length; i += 1) {
                    const prefix = prepend === 'always' || (prepend === 'first' && i === 0);
                    const transformed = `${prefix ? ' ' : ''}${pieces[i]}`.replaceAll(' ', replacement);
                    // Metaspace splits at the replacement character, retaining it
                    // with the following segment (not with the preceding segment).
                    let current = '';
                    for (const ch of transformed) {
                        if (ch === replacement && current) {
                            next.push(current);
                            current = '';
                        }
                        current += ch;
                    }
                    if (current) { next.push(current); }
                }
                pieces = next;
            }
        }
        return pieces;
    }

    encode(text) {
        const ids = [];
        for (const piece of this.preTokenize(text)) {
            ids.push(...this.encodePiece(piece));
        }
        return ids;
    }

    encodePiece(normalized) {
        let symbols = [...normalized];
        // Apply lowest-rank adjacent merge repeatedly (equivalent to ranked BPE).
        while (symbols.length > 1) {
            let best = Infinity;
            let index = -1;
            for (let i = 0; i < symbols.length - 1; i += 1) {
                const rank = this.merges.get(JSON.stringify([symbols[i], symbols[i + 1]]));
                if (rank !== undefined && rank < best) {
                    best = rank;
                    index = i;
                }
            }
            if (index < 0) { break; }
            symbols.splice(index, 2, symbols[index] + symbols[index + 1]);
        }
        const ids = [];
        for (let i = 0; i < symbols.length; i += 1) {
            const piece = `${i ? this.continuingSubwordPrefix : ''}${symbols[i]}${i === symbols.length - 1 ? this.endOfWordSuffix : ''}`;
            const id = this.vocab.get(piece);
            if (id !== undefined) {
                ids.push(id);
                continue;
            }
            if (this.byteFallback) {
                const bytes = new TextEncoder().encode(symbols[i]);
                for (const byte of bytes) {
                    const key = `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`;
                    const byteId = this.vocab.get(key);
                    if (byteId === undefined) { throw new Error(`Missing byte fallback token ${key}`); }
                    ids.push(byteId);
                }
            } else if (this.unknown && this.vocab.has(this.unknown)) {
                ids.push(this.vocab.get(this.unknown));
            } else {
                throw new Error(`No BPE token for ${JSON.stringify(symbols[i])}`);
            }
        }
        return ids;
    }
}
