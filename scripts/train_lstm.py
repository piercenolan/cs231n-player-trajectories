#!/usr/bin/env python3
"""Train LSTM trajectory forecaster on exported tensors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trajectory_lstm import TrajectoryLSTM, masked_mse
from utils.datasets import lstm_out_dir
from utils.lstm_dataset import build_dataloaders


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask_y = batch["mask_y"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = masked_mse(pred, y, mask_y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask_y = batch["mask_y"].to(device)
        pred = model(x)
        loss = masked_mse(pred, y, mask_y)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM trajectory model")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument("--tensor-mode", choices=("single", "multi"), default="multi")
    parser.add_argument(
        "--split",
        choices=("held_out_seed", "temporal_all", "all_seeds"),
        default="held_out_seed",
        help="held_out_seed: train 10s+15s, val 0s (default). temporal_all: 80/20 windows on all seeds.",
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or lstm_out_dir(args.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, scale, split_info = build_dataloaders(
        dataset=args.dataset,
        tensor_mode=args.tensor_mode,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        split=args.split,
    )

    sample = next(iter(train_loader))
    num_players = sample["x"].shape[2]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = TrajectoryLSTM(
        num_players=num_players,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pred_len=args.pred_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on {device} | train batches={len(train_loader)} val={len(val_loader)}")
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device)
        va = eval_epoch(model, val_loader, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}  train={tr:.6f}  val={va:.6f}")

    config = {
        "dataset": args.dataset,
        "tensor_mode": args.tensor_mode,
        "split": args.split,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_players": num_players,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "scale": scale.tolist(),
        "split_info": split_info,
    }

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        ckpt_path,
    )

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
