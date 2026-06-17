import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize():
    print("Quantizing model.onnx to 8-bit integers...")
    quantize_dynamic(
        model_input="model.onnx",
        model_output="model_quant.onnx",
        weight_type=QuantType.QUInt8
    )
    
    orig_size = os.path.getsize("model.onnx") / (1024 * 1024)
    quant_size = os.path.getsize("model_quant.onnx") / (1024 * 1024)
    print(f"Original model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print("Quantization complete! Saved as model_quant.onnx")

if __name__ == "__main__":
    quantize()
