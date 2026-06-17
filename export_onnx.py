import torch
from model import LLM

def export():
    print("Loading checkpoint...")
    checkpoint = torch.load("model.pt", map_location="cpu")
    model = LLM(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        n_layer=checkpoint["n_layer"],
        n_head=checkpoint["n_head"],
        d_ff=checkpoint["d_ff"],
        max_seq_len=checkpoint["max_seq_len"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Exporting model to ONNX format...")
    # Dummy input of shape [1, 10]
    dummy_input = torch.randint(0, checkpoint["vocab_size"], (1, 10), dtype=torch.long)
    
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=14, # stable opset version
        do_constant_folding=True,
        input_names=["idx"],
        output_names=["logits"],
        dynamic_axes={
            "idx": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    print("Export successful! Saved model to model.onnx")

if __name__ == "__main__":
    export()
