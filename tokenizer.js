class BPETokenizer {
    constructor() {
        this.vocab = {};
        this.merges = new Map();
        this.cache = new Map();
        this.wordPattern = /'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+/g;
        
        // Initialize base bytes (0-255)
        for (let i = 0; i < 256; i++) {
            this.vocab[i] = new Uint8Array([i]);
        }
        // Special token ID 256: <|endoftext|>
        this.vocab[256] = new TextEncoder().encode("<|endoftext|>");
    }

    load(data) {
        this.vocabSize = data.vocab_size;
        this.merges.clear();
        for (const [key, val] of Object.entries(data.merges)) {
            const [p1, p2] = key.split(",").map(Number);
            this.merges.set(`${p1},${p2}`, val);
        }

        // Reconstruct vocab in order of token ID
        const sortedMerges = Object.entries(data.merges)
            .map(([k, v]) => ({ pair: k.split(",").map(Number), val: v }))
            .sort((a, b) => a.val - b.val);

        for (const merge of sortedMerges) {
            const [p1, p2] = merge.pair;
            const b1 = this.vocab[p1];
            const b2 = this.vocab[p2];
            if (b1 && b2) {
                const newBytes = new Uint8Array(b1.length + b2.length);
                newBytes.set(b1);
                newBytes.set(b2, b1.length);
                this.vocab[merge.val] = newBytes;
            }
        }
        this.cache.clear();
    }

    _getStats(ids) {
        const stats = new Map();
        for (let i = 0; i < ids.length - 1; i++) {
            const pair = `${ids[i]},${ids[i+1]}`;
            stats.set(pair, (stats.get(pair) || 0) + 1);
        }
        return stats;
    }

    _merge(ids, pair, idx) {
        const newids = [];
        let i = 0;
        const [p1, p2] = pair;
        while (i < ids.length) {
            if (i < ids.length - 1 && ids[i] === p1 && ids[i+1] === p2) {
                newids.push(idx);
                i += 2;
            } else {
                newids.push(ids[i]);
                i += 1;
            }
        }
        return newids;
    }

    _encodeWord(word) {
        if (this.cache.has(word)) {
            return this.cache.get(word);
        }

        const encoder = new TextEncoder();
        let wordBytes = Array.from(encoder.encode(word));
        
        while (wordBytes.length >= 2) {
            const stats = this._getStats(wordBytes);
            let pairToMerge = null;
            let minIdx = Infinity;
            
            for (const pair of stats.keys()) {
                if (this.merges.has(pair)) {
                    const idx = this.merges.get(pair);
                    if (idx < minIdx) {
                        minIdx = idx;
                        pairToMerge = pair;
                    }
                }
            }
            if (pairToMerge === null) {
                break;
            }
            const pair = pairToMerge.split(",").map(Number);
            wordBytes = this._merge(wordBytes, pair, this.merges.get(pairToMerge));
        }

        this.cache.set(word, wordBytes);
        return wordBytes;
    }

    encode(text) {
        const parts = text.split("<|endoftext|>");
        const encodedParts = [];
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            if (part) {
                const matches = part.match(this.wordPattern) || [];
                for (const w of matches) {
                    encodedParts.push(...this._encodeWord(w));
                }
            }
            if (i < parts.length - 1) {
                encodedParts.push(256);
            }
        }
        return encodedParts;
    }

    decode(ids) {
        const decoder = new TextDecoder("utf-8", { fatal: false });
        const parts = [];
        for (const idx of ids) {
            if (this.vocab[idx]) {
                parts.push(this.vocab[idx]);
            } else {
                parts.push(new Uint8Array([idx % 256]));
            }
        }
        
        let totalLen = parts.reduce((acc, curr) => acc + curr.length, 0);
        let result = new Uint8Array(totalLen);
        let offset = 0;
        for (const part of parts) {
            result.set(part, offset);
            offset += part.length;
        }
        return decoder.decode(result);
    }
}

// Export tokenizer for usage in index/main script
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BPETokenizer;
} else {
    window.BPETokenizer = BPETokenizer;
}
