#!/usr/bin/env python3
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from voxel_frontier_features import FRONTIER_FEATURE_NAMES


class FrontierDataset(Dataset):
    def __init__(self, path):
        self.x = []
        self.y = []
        self.groups = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.x.append([float(row[name]) for name in FRONTIER_FEATURE_NAMES])
                self.y.append(float(row["target_score"]))
                self.groups.append((
                    row["map_name"],
                    row.get("target_name", "red_chair"),
                    int(row["state_step"]),
                ))
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)
        print(f"loaded {path}: rows={len(self.x)}, input_dim={self.x.shape[1]}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx


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


class VoxelFrontierSsmNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SsmBlock(hidden_dim) for _ in range(layers)])
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.score_head(x)


def fractional_subset(ds, fraction, seed):
    if fraction >= 0.999:
        return ds
    n = max(1, int(len(ds) * fraction))
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    print(f"data efficiency mode: using {n}/{len(ds)} rows ({fraction:.2%})")
    return Subset(ds, indices[:n])


def evaluate(model, dataset, loader, device):
    model.eval()
    total = 0
    mse_sum = 0.0
    pred_by_idx = {}
    with torch.no_grad():
        for x, y, indices in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y, reduction="sum")
            mse_sum += loss.item()
            total += y.numel()
            for idx, value in zip(indices.tolist(), pred.squeeze(1).cpu().tolist()):
                pred_by_idx[idx] = value

    grouped = defaultdict(list)
    for idx, group in enumerate(dataset.groups):
        grouped[group].append(idx)

    correct = 0
    eligible = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        best_true = max(indices, key=lambda i: float(dataset.y[i].item()))
        best_pred = max(indices, key=lambda i: float(pred_by_idx.get(i, -1e9)))
        correct += int(best_true == best_pred)
        eligible += 1
    return mse_sum / max(total, 1), correct / max(eligible, 1)


def train(args):
    base_train = FrontierDataset(args.train)
    train_ds = fractional_subset(base_train, args.train_fraction, args.seed)
    val_ds = FrontierDataset(args.val)
    input_dim = len(FRONTIER_FEATURE_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = VoxelFrontierSsmNet(input_dim, args.hidden_dim, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_top1 = -1.0
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as f:
        f.write("epoch,train_fraction,train_mse,val_mse,val_top1,best_top1\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        mse_sum = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y)
            loss.backward()
            opt.step()
            mse_sum += F.mse_loss(pred.detach(), y, reduction="sum").item()
            total += y.numel()
        train_mse = mse_sum / max(total, 1)
        val_mse, val_top1 = evaluate(model, val_ds, val_loader, device)
        if val_top1 > best_top1:
            best_top1 = val_top1
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "checkpoint_type": "frontier_scorer",
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "feature_names": FRONTIER_FEATURE_NAMES,
                "train_fraction": args.train_fraction,
            }, args.out)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train_mse={train_mse:.5f} "
                f"val_mse={val_mse:.5f} val_top1={val_top1:.3f} best_top1={best_top1:.3f}"
            )
        with open(args.log, "a") as f:
            f.write(f"{epoch},{args.train_fraction},{train_mse},{val_mse},{val_top1},{best_top1}\n")
    print(f"training finished. best frontier scorer: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Train the initial one-time SSM-style frontier scorer.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
