#!/usr/bin/env python3
"""Train LSTM trajectory forecaster (plain / rule_features / graph)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trajectory_graph_lstm import TrajectoryGraphLSTM
from models.trajectory_lstm import RuleConditionedLSTM, TrajectoryLSTM, masked_mse
from utils.datasets import lstm_out_dir
from utils.lstm_dataset import build_dataloaders
from utils.rule_features import RULE_FEATURE_DIM


def build_model(model_name, num_players, pred_len, hidden_dim, num_layers, rule_feature_dim):
    if model_name == "plain":
        return TrajectoryLSTM(
            num_players=num_players,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pred_len=pred_len,
        )
    if model_name == "rule_features":
        return RuleConditionedLSTM(
            num_players=num_players,
            rule_feature_dim=rule_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pred_len=pred_len,
        )
    if model_name == "graph":
        return TrajectoryGraphLSTM(
            num_players=num_players,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pred_len=pred_len,
        )
    raise ValueError(f"Unknown model: {model_name}")


def rule_penalty_loss(pred, mask_y, scale, max_speed_px=50.0, min_spacing_px=35.0):
    """
    Soft physical constraints on predicted positions (A1b).
    Vectorized velocity cap + batched pairwise spacing per timestep.
    """
    scale_t = scale.view(1, 1, 1, 2).to(pred.device)
    pred_px = pred * scale_t
    total = pred.new_tensor(0.0)
    n_terms = 0

    if pred.shape[1] >= 2:
        vel = pred_px[:, 1:] - pred_px[:, :-1]
        speed = torch.sqrt((vel**2).sum(dim=-1) + 1e-8)
        m = mask_y[:, 1:] & mask_y[:, :-1]
        if m.any():
            pen = torch.relu(speed - max_speed_px)
            total = total + (pen * m.float()).sum() / m.float().sum().clamp(min=1.0)
            n_terms += 1

    B, L, P, _ = pred_px.shape
    for t in range(L):
        m_t = mask_y[:, t]
        if m_t.sum() < 2:
            continue
        pts = pred_px[:, t]
        diff = pts.unsqueeze(2) - pts.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(-1) + 1e-8)
        pair_mask = m_t.unsqueeze(2) & m_t.unsqueeze(1)
        eye = torch.eye(P, device=pred.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        if pair_mask.any():
            pen = torch.relu(min_spacing_px - dist)
            total = total + (pen * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1.0)
            n_terms += 1

    return total if n_terms else pred.new_tensor(0.0)


def forward_batch(model, model_name, batch, device, scale=None, rule_loss_weight=0.0):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    mask_y = batch["mask_y"].to(device)
    if model_name == "rule_features":
        rules = batch["rules_obs"].to(device)
        pred = model(x, rules)
    elif model_name == "graph":
        mask_x = batch["mask_x"].to(device)
        pred = model(x, mask_x)
    else:
        pred = model(x)
    loss = masked_mse(pred, y, mask_y)
    if rule_loss_weight > 0 and scale is not None:
        loss = loss + rule_loss_weight * rule_penalty_loss(pred, mask_y, scale)
    return loss


def train_epoch(model, model_name, loader, optimizer, device, scale, rule_loss_weight):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = forward_batch(
            model, model_name, batch, device, scale=scale, rule_loss_weight=rule_loss_weight
        )
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, model_name, loader, device, scale, rule_loss_weight):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss = forward_batch(
            model, model_name, batch, device, scale=scale, rule_loss_weight=rule_loss_weight
        )
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM trajectory model")
    parser.add_argument("--dataset", default="sportsmot_example")
    parser.add_argument(
        "--model",
        choices=("plain", "rule_features", "graph"),
        default="plain",
    )
    parser.add_argument("--tensor-mode", choices=("single", "multi"), default="multi")
    parser.add_argument(
        "--split",
        choices=("held_out_seed", "temporal_all", "all_seeds"),
        default="temporal_all",
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--rule-loss-weight",
        type=float,
        default=0.0,
        help="A1b: weight for velocity/spacing penalty on predictions (rule_features only)",
    )
    parser.add_argument("--val-seed", default=None, help="Held-out seed for held_out_seed split")
    args = parser.parse_args()

    require_rules = args.model == "rule_features"
    default_sub = f"lstm_{args.model}"
    out_dir = Path(args.out_dir or (lstm_out_dir(args.dataset) / default_sub))
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_kw = dict(
        dataset=args.dataset,
        tensor_mode=args.tensor_mode,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        split=args.split,
        require_rule_features=require_rules,
    )
    if args.val_seed:
        dl_kw["val_seed"] = args.val_seed
    train_loader, val_loader, scale, split_info = build_dataloaders(**dl_kw)

    scale_t = torch.tensor(scale, dtype=torch.float32)
    rule_w = args.rule_loss_weight if args.model == "rule_features" else 0.0

    sample = next(iter(train_loader))
    num_players = sample["x"].shape[2]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(
        args.model,
        num_players,
        args.pred_len,
        args.hidden_dim,
        args.num_layers,
        RULE_FEATURE_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": []}

    print(
        f"Training {args.model} on {device} | "
        f"train={len(train_loader)} val={len(val_loader)}"
    )
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, args.model, train_loader, optimizer, device, scale_t, rule_w)
        va = eval_epoch(model, args.model, val_loader, device, scale_t, rule_w)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}  train={tr:.6f}  val={va:.6f}")

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "tensor_mode": args.tensor_mode,
        "split": args.split,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_players": num_players,
        "rule_feature_dim": RULE_FEATURE_DIM,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "scale": scale.tolist(),
        "split_info": split_info,
        "rule_loss_weight": rule_w,
    }

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
