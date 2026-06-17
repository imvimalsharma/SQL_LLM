import json
import os
import re

class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.merges = {}  # (p1, p2) -> parent
        self.vocab = {}   # token_id -> bytes
        self.special_tokens = {"<|endoftext|>": 256}
        self.inverse_special_tokens = {256: "<|endoftext|>"}
        self.cache = {}   # word -> token_ids cache for fast encoding
        
        # GPT-2 style regex for splitting text into words/tokens
        self.word_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+""")
        
        # Initialize vocab with base bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
        self.vocab[256] = b"<|endoftext|>"

    def _get_stats(self, ids):
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, vocab_size, verbose=True):
        self.vocab_size = vocab_size
        # Filter out special tokens first
        cleaned_text = text.replace("<|endoftext|>", "")
        
        # Train BPE on list of words to keep it extremely fast
        words = self.word_pattern.findall(cleaned_text)
        word_ids = [list(w.encode("utf-8")) for w in words]
        
        num_merges = vocab_size - 257 # 256 basic bytes + 1 special token = 257
        
        for i in range(num_merges):
            stats = {}
            for ids in word_ids:
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
                    
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get)
            new_id = 257 + i
            
            # Merge in all word representations
            word_ids = [self._merge(ids, best_pair, new_id) for ids in word_ids]
            self.merges[best_pair] = new_id
            
            p0, p1 = best_pair
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
            if verbose and (i + 1) % 500 == 0:
                print(f"BPE Merge {i+1}/{num_merges}: merged {best_pair} -> {new_id} ({self.vocab[new_id].decode('utf-8', errors='replace')!r})")
                
        # Clear cache since merges changed
        self.cache = {}

    def _encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
            
        word_bytes = list(word.encode("utf-8"))
        while len(word_bytes) >= 2:
            stats = self._get_stats(word_bytes)
            pair_to_merge = None
            min_idx = float('inf')
            for pair in stats:
                if pair in self.merges:
                    idx = self.merges[pair]
                    if idx < min_idx:
                        min_idx = idx
                        pair_to_merge = pair
            if pair_to_merge is None:
                break
            word_bytes = self._merge(word_bytes, pair_to_merge, self.merges[pair_to_merge])
            
        self.cache[word] = word_bytes
        return word_bytes

    def encode(self, text):
        parts = text.split("<|endoftext|>")
        encoded_parts = []
        for i, part in enumerate(parts):
            if part:
                words = self.word_pattern.findall(part)
                for w in words:
                    encoded_parts.extend(self._encode_word(w))
            if i < len(parts) - 1:
                encoded_parts.append(256)
        return encoded_parts

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                part_bytes.append(bytes([idx % 256]))
        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def save(self, file_path):
        serialized_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": serialized_merges
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v
            
        self.vocab = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        self.vocab[256] = b"<|endoftext|>"
        
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p1, p2), v in sorted_merges:
            self.vocab[v] = self.vocab[p1] + self.vocab[p2]
            
        self.cache = {}

if __name__ == "__main__":
    print("Testing BPETokenizer...")
    text = "[INST] Schema: CREATE TABLE t1 (c1 INT) | Question: show c1 [/INST] SQL: SELECT c1 FROM t1 <|endoftext|>"
    tok = BPETokenizer(vocab_size=300)
    tok.train(text * 10, vocab_size=300, verbose=False)
    
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    
    print("Original text:", text)
    print("Encoded IDs:", encoded)
    print("Decoded text:", decoded)
    assert decoded == text, "Decoded text does not match original!"
    print("Tokenizer test passed successfully!")
