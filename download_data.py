import os
import json
import requests
import random
from tqdm import tqdm

def download_dataset(url, json_path):
    if not os.path.exists(json_path):
        print(f"Downloading {url}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="sql_create_context_v4.json"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"Dataset already exists at {json_path}")

def format_data(json_path, train_path, val_path):
    print("Formatting dataset into instruction-tuning text...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_examples = []
    for item in data:
        context = item.get("context", "").strip()
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        
        # Format as standard instruction tuning example for decoder-only model
        formatted = f"[INST] Schema: {context} | Question: {question} [/INST] SQL: {answer} <|endoftext|>\n"
        formatted_examples.append(formatted)
        
    # Shuffle to mix up tables and schemas
    random.seed(42)
    random.shuffle(formatted_examples)
    
    # Split: 90% train, 10% validation
    split_idx = int(len(formatted_examples) * 0.90)
    train_data = formatted_examples[:split_idx]
    val_data = formatted_examples[split_idx:]
    
    # Write to files
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
        
    total_chars = sum(len(x) for x in formatted_examples)
    print(f"Formatted {len(formatted_examples)} examples.")
    print(f"Saved {len(train_data)} train examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    print(f"Total dataset size: {total_chars / 1024 / 1024:.2f} MB (approx {total_chars / 4 / 1000000:.2f} Million tokens)")

if __name__ == "__main__":
    url = "https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json"
    data_dir = "data"
    json_path = os.path.join(data_dir, "sql_create_context_v4.json")
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")
    
    download_dataset(url, json_path)
    format_data(json_path, train_path, val_path)
