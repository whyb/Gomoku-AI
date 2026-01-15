import argparse
import torch
from model_dy import GomokuNetDyn, load_model_if_exists

# 兼容旧版 torch 的 pytree 接口，避免 transformers 依赖报错
try:
    from torch.utils import _pytree as _pytree
    if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
        def _register_wrapper(cls, flatten_fn, unflatten_fn, **kwargs):
            return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
        _pytree.register_pytree_node = _register_wrapper
except Exception:
    pass

def export(model_path, board_size, onnx_path, torchscript_path):
    model = GomokuNetDyn()
    load_model_if_exists(model, model_path)
    model.eval()
    dummy = torch.randn(1, 2, board_size, board_size)
    if onnx_path:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "policy_logits": {1: "cells"}
            },
            opset_version=18,
            external_data=False,
        )
        print(f"ONNX saved: {onnx_path}")
    if torchscript_path:
        traced = torch.jit.trace(model, dummy)
        traced.save(torchscript_path)
        print(f"TorchScript saved: {torchscript_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--board_size", type=int, required=True)
    parser.add_argument("--onnx_path", type=str, default="model_dy.onnx")
    parser.add_argument("--torchscript_path", type=str, default="model_dy.pt")
    args = parser.parse_args()
    export(args.model_path, args.board_size, args.onnx_path, args.torchscript_path)
