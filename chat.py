import os
import sys
import torch
import torch.nn.functional as F
from tokenizer import BPETokenizer
from model import LLM

# Color codes for premium CLI styling
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def load_model(checkpoint_path="model.pt", tokenizer_path="data/tokenizer.json"):
    if not os.path.exists(checkpoint_path):
        print(f"{YELLOW}Error: Model checkpoint '{checkpoint_path}' not found.{RESET}")
        print("Please train the model first by running: python3 train.py")
        sys.exit(1)
        
    print(f"{CYAN}Loading custom BPE Tokenizer from {tokenizer_path}...{RESET}")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"{CYAN}Loading model checkpoint from {checkpoint_path}...{RESET}")
    # Load model configuration and weights
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = LLM(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        n_layer=checkpoint["n_layer"],
        n_head=checkpoint["n_head"],
        d_ff=checkpoint["d_ff"],
        max_seq_len=checkpoint["max_seq_len"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"{GREEN}Success! Model loaded on {device.upper()}. (Step {checkpoint['step']}, Best Val Loss: {checkpoint['best_val_loss']:.4f}){RESET}")
    return model, tokenizer, device

@torch.no_grad()
def generate_sql(model, tokenizer, device, schema, question, temp=0.2, top_k=20):
    # Construct standard instruction prompt
    prompt = f"[INST] Schema: {schema} | Question: {question} [/INST] SQL:"
    
    # Encode prompt
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print(f"\n{GREEN}{BOLD}Generated SQL:{RESET} ", end="", flush=True)
    
    generated_ids = []
    # Generate up to 128 tokens
    for _ in range(128):
        # Crop context to max_seq_len
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
        
        # Forward pass
        logits, _ = model(idx_cond) # shape [B, 1, vocab_size]
        logits = logits[:, -1, :]
        
        # Scale by temperature (temperature=0.2 is best for high accuracy, deterministic SQL)
        if temp > 0:
            logits = logits / temp
            
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
        # Sample or argmax
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for end of text token
        token_id = next_token.item()
        if token_id == 256: # <|endoftext|>
            break
            
        # Append to context
        idx = torch.cat((idx, next_token), dim=1)
        
        # Decode and stream print
        token_str = tokenizer.decode([token_id])
        print(f"{BLUE}{token_str}{RESET}", end="", flush=True)
    print("\n")

def main():
    print(f"{CYAN}{BOLD}")
    print("=" * 60)
    print("      🧠 LOCAL TEXT-TO-SQL LLM (8.4M LLaMA-style) 🧠      ")
    print("               Trained from scratch 100% locally        ")
    print("=" * 60)
    print(f"{RESET}")
    
    model, tokenizer, device = load_model()
    
    # Default schema to start with
    default_schema = "CREATE TABLE users (id INT, name VARCHAR, age INT, department VARCHAR, salary INT)"
    print(f"\n{YELLOW}Active Schema:{RESET} {default_schema}")
    print(f"{CYAN}(Press Enter to keep this schema, or type 'schema [your CREATE TABLE statement]' to change it. Type 'exit' to quit.){RESET}")
    
    schema = default_schema
    
    while True:
        try:
            user_input = input(f"\n{BOLD}Question > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print(f"{YELLOW}Goodbye!{RESET}")
                break
                
            # Check if user wants to change schema
            if user_input.lower().startswith("schema "):
                new_schema = user_input[7:].strip()
                if new_schema:
                    schema = new_schema
                    print(f"{GREEN}Schema updated to:{RESET} {schema}")
                continue
                
            # Generate SQL
            generate_sql(model, tokenizer, device, schema, user_input)
            
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Exiting...{RESET}")
            break

if __name__ == "__main__":
    main()
