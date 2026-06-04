#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PATCH_SIZE = 25

class GridPolicyDataset(Dataset):
    def __init__(self, path):
        self.x = []
        self.y_action = []
        self.y_reward = []

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = []

                for i in range(PATCH_SIZE):
                    feat.append(float(row[f"patch_{i}"]))

                feat += [
                    float(row["robot_r"]),
                    float(row["robot_c"]),
                    float(row["prev_action"]),
                    float(row["observed_nodes"]),
                    float(row["observed_edges"]),
                    float(row["frontier_count"]),
                    float(row["visited_count"]),
                ]

                rewards = [
                    float(row["reward_forward"]),
                    float(row["reward_right"]),
                    float(row["reward_down"]),
                    float(row["reward_left"]),
                    float(row["reward_stop"]),
                ]

                self.x.append(feat)
                self.y_action.append(int(row["best_reward_action"]))
                self.y_reward.append(rewards)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y_action = torch.tensor(self.y_action, dtype=torch.long)
        self.y_reward = torch.tensor(self.y_reward, dtype=torch.float32)

        print(f"loaded {path}: rows={len(self.x)}, input_dim={self.x.shape[1]}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_action[idx], self.y_reward[idx]

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
        return self.action_head(x), self.reward_head(x)

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y_action, y_reward in loader:
            x = x.to(device)
            y_action = y_action.to(device)
            y_reward = y_reward.to(device)

            logits, reward_pred = model(x)
            loss_action = F.cross_entropy(logits, y_action)
            loss_reward = F.mse_loss(reward_pred, y_reward)
            loss = loss_action + 0.2 * loss_reward

            pred = logits.argmax(dim=1)
            correct += (pred == y_action).sum().item()
            total += y_action.numel()
            loss_sum += loss.item() * y_action.numel()

    return loss_sum / max(total, 1), correct / max(total, 1)

def train(args):
    train_ds = GridPolicyDataset(args.train)
    val_ds = GridPolicyDataset(args.val)

    input_dim = train_ds.x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GridPolicyNet(input_dim=input_dim, hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,best_acc\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y_action, y_reward in train_loader:
            x = x.to(device)
            y_action = y_action.to(device)
            y_reward = y_reward.to(device)

            opt.zero_grad()
            logits, reward_pred = model(x)

            loss_action = F.cross_entropy(logits, y_action)
            loss_reward = F.mse_loss(reward_pred, y_reward)
            loss = loss_action + 0.2 * loss_reward

            loss.backward()
            opt.step()

            pred = logits.argmax(dim=1)
            correct += (pred == y_action).sum().item()
            total += y_action.numel()
            loss_sum += loss.item() * y_action.numel()

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "actions": 5,
            }, args.out)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} best_acc={best_acc:.3f}")

        with open(args.log, "a") as f:
            f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{best_acc}\n")

    print(f"training finished. best model: {args.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
