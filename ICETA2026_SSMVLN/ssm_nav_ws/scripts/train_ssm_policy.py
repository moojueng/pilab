#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


LEGACY_FEATURE_COLUMNS = [
    "start_id",
    "target_id",
    "path_length",
    "vision_mean",
    "vision_dark",
    "vision_edge",
]

ACTION_FEATURE_COLUMNS = [
    "current_x",
    "current_y",
    "goal_x",
    "goal_y",
    "delta_goal_x",
    "delta_goal_y",
    "distance_to_goal",
    "vision_mean",
    "vision_dark",
    "vision_edge",
    "front_clear",
    "left_clear",
    "right_clear",
    "min_range",
]

ROBOT_FRAME_FEATURE_COLUMNS = [
    "current_x",
    "current_y",
    "goal_x",
    "goal_y",
    "robot_dir_x",
    "robot_dir_y",
    "distance_to_goal",
    "vision_mean",
    "vision_dark",
    "vision_edge",
    "front_clear",
    "left_clear",
    "right_clear",
    "min_range",
]


class NavCsvDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.rows = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames or []
            for row in reader:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"Empty dataset: {csv_path}")

        self.feature_columns = self.detect_feature_columns()
        self.label_column = self.detect_label_column()

        self.x = []
        self.y = []

        for row in self.rows:
            features = [float(row[col]) for col in self.feature_columns]
            label = int(float(row[self.label_column]))
            self.x.append(features)
            self.y.append(label)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        self.input_dim = self.x.shape[1]
        self.num_classes = int(self.y.max().item()) + 1

        print(
            f"Loaded {csv_path}: rows={len(self.rows)}, "
            f"input_dim={self.input_dim}, classes={self.num_classes}, "
            f"label={self.label_column}"
        )

    def detect_feature_columns(self):
        fields = set(self.fieldnames)

        if all(col in fields for col in ROBOT_FRAME_FEATURE_COLUMNS):
            return ROBOT_FRAME_FEATURE_COLUMNS

        if all(col in fields for col in ACTION_FEATURE_COLUMNS):
            return ACTION_FEATURE_COLUMNS

        if all(col in fields for col in LEGACY_FEATURE_COLUMNS):
            return LEGACY_FEATURE_COLUMNS

        missing_robot = [c for c in ROBOT_FRAME_FEATURE_COLUMNS if c not in fields]
        missing_action = [c for c in ACTION_FEATURE_COLUMNS if c not in fields]
        raise ValueError(
            f"Could not detect supported feature columns in {self.csv_path}\n"
            f"fields={self.fieldnames}\n"
            f"missing_robot_frame={missing_robot}\n"
            f"missing_action={missing_action}"
        )

    def detect_label_column(self):
        fields = set(self.fieldnames)

        if "teacher_action" in fields:
            return "teacher_action"

        if "teacher_next_node" in fields:
            return "teacher_next_node"

        raise ValueError(f"Missing label column in {self.csv_path}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SsmBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.numel()

    return total_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(path, model, args, input_dim, num_classes, feature_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_classes": num_classes,
            "feature_columns": feature_columns,
        },
        path,
    )


def train(args):
    train_dataset = NavCsvDataset(args.train)
    val_dataset = NavCsvDataset(args.val)

    if train_dataset.input_dim != val_dataset.input_dim:
        raise ValueError("Train/val input_dim mismatch")

    input_dim = train_dataset.input_dim
    num_classes = max(train_dataset.num_classes, val_dataset.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SsmPolicyNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,train_acc,val_loss,val_acc,best_val_acc\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.numel()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.out,
                model,
                args,
                input_dim,
                num_classes,
                train_dataset.feature_columns,
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            line = (
                f"epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.3f} "
                f"best_val_acc={best_val_acc:.3f}"
            )
            print(line)

        with open(log_path, "a") as log_file:
            log_file.write(
                f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{best_val_acc}\n"
            )

    print("Training finished.")
    print(f"Best model saved to: {args.out}")
    print(f"Training log saved to: {args.log}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
