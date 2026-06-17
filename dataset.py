import os
import json
import numpy as np
import torch
from tokenizer import BPETokenizer

def prepare_dataset(data_dir="data", vocab_size=4096):
    train_txt = os.path.join(data_dir, "train.txt")
    val_txt = os.path.join(data_dir, "val.txt")
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if not os.path.exists(tokenizer_path):
        print("Training BPE Tokenizer on 100KB subset of training text...")
        with open(train_txt, 'r', encoding='utf-8') as f:
            sample_text = f.read(100 * 1024)
            
        tokenizer.train(sample_text, vocab_size=vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
        
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Tokenizing train.txt...")
        with open(train_txt, 'r', encoding='utf-8') as f:
            train_text = f.read()
        train_ids = tokenizer.encode(train_text)
        train_np = np.array(train_ids, dtype=np.uint16)
        train_np.tofile(train_bin)
        print(f"Saved {len(train_ids)} tokens to {train_bin}")
        
        print("Tokenizing val.txt...")
        with open(val_txt, 'r', encoding='utf-8') as f:
            val_text = f.read()
        val_ids = tokenizer.encode(val_text)
        val_np = np.array(val_ids, dtype=np.uint16)
        val_np.tofile(val_bin)
        print(f"Saved {len(val_ids)} tokens to {val_bin}")
    else:
        print("Tokenized binary files already exist.")
        
    return tokenizer

class SQLDataLoader:
    def __init__(self, data_dir="data", device="cpu"):
        self.device = device
        
        train_np = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
        val_np = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
        
        # Preload the entire tokenized dataset onto the device
        print(f"Preloading dataset tensors onto {device}...")
        self.train_data = torch.from_numpy(train_np.astype(np.int64)).to(device)
        self.val_data = torch.from_numpy(val_np.astype(np.int64)).to(device)
        print(f"Loaded {len(self.train_data)} training tokens and {len(self.val_data)} validation tokens on {device}.")
        
    def get_batch(self, split, batch_size, max_seq_len):
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random start indices on CPU (randint is fast on CPU, avoids device calls)
        ix = torch.randint(len(data) - max_seq_len, (batch_size,)).to(self.device)
        
        # Vectorized gather using broadcasting (avoids list comprehension and device loops)
        arange = torch.arange(max_seq_len, device=self.device)
        indices = ix.unsqueeze(1) + arange.unsqueeze(0) # Shape: [batch_size, max_seq_len]
        
        x = data[indices]
        y = data[indices + 1]
        return x, y

if __name__ == "__main__":
    print("Testing dataset prep...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = prepare_dataset(vocab_size=4096)
    
    loader = SQLDataLoader(device=device)
    x, y = loader.get_batch("train", batch_size=4, max_seq_len=64)
    print("x shape:", x.shape, "device:", x.device)
    print("y shape:", y.shape, "device:", y.device)
    print("Sample x decoded:", tok.decode(x[0].tolist()[:20]))
    print("Dataset test passed successfully!")
