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


class GridPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=3, actions=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SsmBlock(hidden_dim) for _ in range(layers)])
        self.action_head = nn.Linear(hidden_dim, actions)
        self.reward_head = nn.Linear(hidden_dim, actions)

    def forward(self, x):
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        action_logits = self.action_head(x)
        reward_values = self.reward_head(x)
        return action_logits, reward_values


def export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    layers = int(ckpt["layers"])
    actions = int(ckpt["actions"])

    model = GridPolicyNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        actions=actions,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, input_dim, dtype=torch.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["action_logits", "reward_values"],
        dynamic_axes={
            "input": {0: "batch"},
            "action_logits": {0: "batch"},
            "reward_values": {0: "batch"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model: {out}")
    print(f"input_dim={input_dim}, hidden_dim={hidden_dim}, layers={layers}, actions={actions}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export(args)


if __name__ == "__main__":
    main()
