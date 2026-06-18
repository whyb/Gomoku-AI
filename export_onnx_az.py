"""
AlphaZero 模型导出为 ONNX 和 TorchScript

用法:
  # 导出 Small 模型
  python export_onnx_az.py alpaz_small_15x15_model.pth --board_size 15 --model small

  # 导出 Standard 模型
  python export_onnx_az.py alpaz_standard_15x15_model.pth --board_size 15 --model standard

  # 自定义输出路径
  python export_onnx_az.py alpaz_small_15x15_model.pth --board_size 15 --model small --onnx_path my_model.onnx --torchscript_path my_model.pt
"""

import argparse
import torch
from model_alphazero import GomokuNetAlphaZero, GomokuNetAlphaZeroSmall


def export(model_path, board_size, model_type, onnx_path, torchscript_path):
    # 加载模型
    if model_type == 'small':
        model = GomokuNetAlphaZeroSmall()
    else:
        model = GomokuNetAlphaZero()

    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载 checkpoint (step={checkpoint.get('update_step', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ 加载原始权重")
    model.eval()

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"模型: {model_type} ({params:,} 参数)")
    print(f"输入: (1, 2, {board_size}, {board_size})")

    # 构造 dummy 输入
    dummy = torch.randn(1, 2, board_size, board_size)

    # 验证前向传播
    with torch.no_grad():
        policy, value = model(dummy)
    print(f"输出: policy={policy.shape}, value={value.shape}")

    # 导出 ONNX
    if onnx_path:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "policy_logits": {0: "batch", 1: "cells"},
                "value": {0: "batch"},
            },
            opset_version=18,
            external_data=False,
        )
        print(f"ONNX saved: {onnx_path}")

    # 导出 TorchScript
    if torchscript_path:
        traced = torch.jit.trace(model, dummy)
        traced.save(torchscript_path)
        print(f"TorchScript saved: {torchscript_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 AlphaZero 模型为 ONNX/TorchScript")
    parser.add_argument("model_path", help="模型权重文件路径 (.pth)")
    parser.add_argument("--board_size", type=int, required=True, help="棋盘大小")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "standard"], help="模型类型 (默认 small)")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="ONNX 输出路径 (默认自动生成)")
    parser.add_argument("--torchscript_path", type=str, default=None,
                        help="TorchScript 输出路径 (默认自动生成)")
    args = parser.parse_args()

    # 自动生成输出路径
    if args.onnx_path is None:
        args.onnx_path = f"gobang_az_{args.model}_{args.board_size}x{args.board_size}.onnx"
    if args.torchscript_path is None:
        args.torchscript_path = f"gobang_az_{args.model}_{args.board_size}x{args.board_size}.pt"

    export(args.model_path, args.board_size, args.model,
           args.onnx_path, args.torchscript_path)
