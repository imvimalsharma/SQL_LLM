import torch
from tokenizer import BPETokenizer
from model import LLM
from chat import generate_sql

def run_tests():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running test generation on {device}...")
    
    checkpoint = torch.load("model.pt", map_location=device)
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
    
    tokenizer = BPETokenizer()
    tokenizer.load("data/tokenizer.json")
    
    schema = "CREATE TABLE users (id INT, name VARCHAR, age INT, country VARCHAR)"
    
    test_cases = [
        "Find the name of users whose age is greater than 25.",
        "Select the average age of users.",
        "Show the name of users from India.",
        "How many users are in the table?"
    ]
    
    print("\n" + "="*50)
    print("           TEXT-TO-SQL TEST GENERATIONS")
    print("="*50)
    print(f"Schema: {schema}\n")
    
    for q in test_cases:
        print(f"Question: {q}")
        generate_sql(model, tokenizer, device, schema, q, temp=0.1, top_k=5)
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
