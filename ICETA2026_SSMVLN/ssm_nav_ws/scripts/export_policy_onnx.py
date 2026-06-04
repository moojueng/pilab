#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SsmBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        gate = torch.sigmoid(self.gate_proj(x))
        x = F.silu(self.in_proj(x)) * gate
        x = self.out_proj(x)
        return self.norm(x + residual)


class SsmPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SsmBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def export(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    input_dim = int(checkpoint.get("input_dim", 14))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    num_layers = int(checkpoint.get("num_layers", 3))

    if "num_classes" in checkpoint:
        num_classes = int(checkpoint["num_classes"])
    elif "num_actions" in checkpoint:
        num_classes = int(checkpoint["num_actions"])
    else:
        num_classes = 4

    model = SsmPolicyNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model: {output_path}")
    print(f"input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
