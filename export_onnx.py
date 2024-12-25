import os
import sys
import torch
from model import GomokuNetV2

BOARD_SIZE = 8  # 定义棋盘大小

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Loaded model weights from {file_path}")
    else:
        raise FileNotFoundError(f"No saved model weights found at {file_path}")

def export_torchscript(model, output_path):
    # 将模型转换为 TorchScript
    model.eval()
    example_input = torch.randn(1, BOARD_SIZE * BOARD_SIZE)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)
    print(f"TorchScript model saved to {output_path}")

def export_onnx(model, output_path):
    # 将模型转换为 ONNX
    model.eval()
    example_input = torch.randn(1, BOARD_SIZE * BOARD_SIZE)
    torch.onnx.export(model, example_input, output_path,
                      export_params=True, opset_version=10, 
                      do_constant_folding=True, input_names=['input'], 
                      output_names=['output'])
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    pth_file_path = sys.argv[1]  # 输入pth权重文件路径
    torchscript_output_path = "model_torchscript.pt"  # 输出 TorchScript 模型文件路径
    onnx_output_path = "model.onnx"  # 输出 ONNX 模型文件路径

    model = GomokuNetV2()
    load_model(model, pth_file_path)
    
    export_torchscript(model, torchscript_output_path)
    export_onnx(model, onnx_output_path)
