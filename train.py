import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from model import LLM
from dataset import SQLDataLoader, prepare_dataset

# Hyperparameters
vocab_size = 4096
d_model = 256
n_layer = 6
n_head = 8
d_ff = 1024
max_seq_len = 256

batch_size = 32
learning_rate = 1e-3
min_lr = 1e-4
weight_decay = 0.01
max_iters = 2500
eval_interval = 250
eval_iters = 50
warmup_iters = 200

# Device selection
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

# 1. Prepare data and tokenizer
print("Loading tokenizer and preparing dataset binary files...")
tokenizer = prepare_dataset(vocab_size=vocab_size)
loader = SQLDataLoader(device=device)

# 2. Instantiate Model
print("Initializing model...")
model = LLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layer=n_layer,
    n_head=n_head,
    d_ff=d_ff,
    max_seq_len=max_seq_len
)
model.to(device)

# Count parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized. Trainable parameters: {params / 1e6:.3f} Million")

# 3. Setup optimizer
# Filter out weight decay for 1D params like biases or LayerNorms
decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

# Learning rate schedule helper
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split, batch_size, max_seq_len)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 4. Training loop
checkpoint_path = "model.pt"
start_step = 0
best_val_loss = float("inf")

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resuming training from step {start_step} with best validation loss: {best_val_loss:.4f}")
else:
    print("Starting training from scratch...")

start_time = time.time()

# We will run training for max_iters
for it in range(start_step, max_iters + 1):
    # Determine learning rate
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    # Evaluate periodic loss
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"\nStep {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if it > 0:
                print(f"New best validation loss! Saving model checkpoint to model.pt...")
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "d_ff": d_ff,
                    "max_seq_len": max_seq_len,
                    "best_val_loss": best_val_loss,
                    "step": it
                }
                torch.save(checkpoint, "model.pt")
                
    # Fetch batch
    x, y = loader.get_batch("train", batch_size, max_seq_len)
    
    # Forward and backward pass
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Simple console progress
    if it % 10 == 0:
        elapsed = time.time() - start_time
        tokens_processed = it * batch_size * max_seq_len
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        print(f"Step {it}/{max_iters} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.0f}")

print("\nTraining complete!")
# Save final model
print("Saving final model checkpoint...")
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "d_model": d_model,
    "n_layer": n_layer,
    "n_head": n_head,
    "d_ff": d_ff,
    "max_seq_len": max_seq_len,
    "best_val_loss": loss.item(),
    "step": max_iters
}
torch.save(checkpoint, "model.pt")
print("Saved final model to model.pt.")
