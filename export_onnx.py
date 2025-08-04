import os
import torch
import argparse
from model import GomokuNetV2
from config import Config, update_config_from_cli

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Loaded model weights from {file_path}")
    else:
        raise FileNotFoundError(f"No saved model weights found at {file_path}")

def export_torchscript(model, output_path, board_size):
    model.eval()
    example_input = torch.randn(1, board_size * board_size)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)
    print(f"TorchScript model saved to {output_path}")

def export_onnx(model, output_path, board_size):
    model.eval()
    example_input = torch.randn(1, board_size * board_size)
    torch.onnx.export(model, example_input, output_path,
                      export_params=True, opset_version=10, 
                      do_constant_folding=True, input_names=['input'], 
                      output_names=['output'])
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the input model .pth file (e.g., gobang_best_model.pth)")
    parser.add_argument("--board_size", type=int, required=True, 
                      help="Size of the game board (must match training configuration)")
    parser.add_argument("--win_condition", type=int, required=True, 
                      help="Number of consecutive stones to win (must match training configuration)")
    parser.add_argument("--onnx_path", type=str, 
                      help="Path to save the ONNX model (e.g., custom_model.onnx)")
    parser.add_argument("--torchscript_path", type=str, 
                      help="Path to save the TorchScript model (e.g., custom_model.pt)")
    
    args = parser.parse_args()
    config = update_config_from_cli(args)
    board_size = config.BOARD_SIZE
    win_condition = config.WIN_CONDITION

    # 打印导出配置信息
    print(f"===== 导出配置 =====")
    print(f"输入模型: {args.model_path}")
    print(f"棋盘尺寸: {board_size}x{board_size}")
    print(f"胜利条件: 连{win_condition}子")
    print(f"===================\n")

    # 初始化模型（使用训练时的棋盘尺寸）
    model = GomokuNetV2(board_size=board_size)
    load_model(model, args.model_path)

    # 处理导出路径（默认路径包含关键参数信息）
    default_onnx = f"models/model_bs{board_size}_win{win_condition}.onnx"
    default_torchscript = f"models/model_torchscript_bs{board_size}_win{win_condition}.pt"
    
    onnx_output_path = args.onnx_path if args.onnx_path else default_onnx
    torchscript_output_path = args.torchscript_path if args.torchscript_path else default_torchscript

    # 导出模型
    export_torchscript(model, torchscript_output_path, board_size)
    export_onnx(model, onnx_output_path, board_size)
    