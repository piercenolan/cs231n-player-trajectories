#!/usr/bin/env python3
"""Train LSTM trajectory forecaster (plain / rule_features / graph)."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trajectory_graph_lstm import TrajectoryGraphLSTM
from models.trajectory_lstm import RuleConditionedLSTM, TrajectoryLSTM, masked_mse
from utils.datasets import lstm_out_dir
from utils.linear_baseline import linear_prediction_norm_torch
from utils.lstm_dataset import build_dataloaders
from utils.lstm_val_metrics import forecast_ade_px_rollout
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


def scheduled_sampling_prob(epoch, ramp_end=40, max_p=0.3):
    if epoch <= 0:
        return 0.0
    if epoch >= ramp_end:
        return max_p
    return max_p * (epoch / ramp_end)


def predict_positions(model, model_name, batch, device, residual=False, pred_len=4):
    """Model output; if residual, add constant-velocity linear baseline."""
    x = batch["x"].to(device)
    if model_name == "rule_features":
        rules = batch["rules_obs"].to(device)
        delta = model(x, rules)
    elif model_name == "graph":
        mask_x = batch["mask_x"].to(device)
        delta = model(x, mask_x)
    else:
        delta = model(x)
    if residual:
        y_lin = linear_prediction_norm_torch(x, pred_len)
        return y_lin + delta
    return delta


def masked_l1_px(pred, target, mask_y, scale):
    """Differentiable L1 in pixel space (closer to ADE than MSE alone)."""
    scale_t = scale.view(1, 1, 1, 2).to(pred.device)
    diff = (pred - target).abs() * scale_t
    mask_f = mask_y.float().unsqueeze(-1).expand_as(diff)
    return (diff * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def apply_scheduled_sampling(model, model_name, batch, device, p, residual=False, pred_len=4):
    if p <= 0 or random.random() > p:
        return batch
    x = batch["x"].to(device).clone()
    with torch.no_grad():
        pred = predict_positions(model, model_name, {**batch, "x": x}, device, residual, pred_len)
    k = min(2, x.shape[1] - 1, pred.shape[1])
    for i in range(k):
        x[:, -(k - i)] = pred[:, i]
    batch = dict(batch)
    batch["x"] = x
    return batch


def forward_batch(
    model,
    model_name,
    batch,
    device,
    scale=None,
    rule_loss_weight=0.0,
    residual=False,
    pred_len=4,
    px_l1_weight=0.0,
):
    y = batch["y"].to(device)
    mask_y = batch["mask_y"].to(device)
    pred = predict_positions(model, model_name, batch, device, residual=residual, pred_len=pred_len)
    loss = masked_mse(pred, y, mask_y)
    if px_l1_weight > 0 and scale is not None:
        loss = loss + px_l1_weight * masked_l1_px(pred, y, mask_y, scale)
    if rule_loss_weight > 0 and scale is not None:
        loss = loss + rule_loss_weight * rule_penalty_loss(pred, mask_y, scale)
    return loss


def train_epoch(
    model,
    model_name,
    loader,
    optimizer,
    device,
    scale,
    rule_loss_weight,
    scheduled_p=0.0,
    residual=False,
    pred_len=4,
    px_l1_weight=0.0,
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        if scheduled_p > 0:
            batch = apply_scheduled_sampling(
                model, model_name, batch, device, scheduled_p, residual, pred_len
            )
        loss = forward_batch(
            model,
            model_name,
            batch,
            device,
            scale=scale,
            rule_loss_weight=rule_loss_weight,
            residual=residual,
            pred_len=pred_len,
            px_l1_weight=px_l1_weight,
        )
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model,
    model_name,
    loader,
    device,
    scale,
    rule_loss_weight,
    residual=False,
    pred_len=4,
    px_l1_weight=0.0,
):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss = forward_batch(
            model,
            model_name,
            batch,
            device,
            scale=scale,
            rule_loss_weight=rule_loss_weight,
            residual=residual,
            pred_len=pred_len,
            px_l1_weight=px_l1_weight,
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
    parser.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="Mix model predictions into observation window during training",
    )
    parser.add_argument(
        "--post-eval",
        action="store_true",
        help="After training, run multi-seed eval and record A1_temporal_all",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Predict correction on top of constant-velocity linear baseline",
    )
    parser.add_argument(
        "--optimize-forecast-ade",
        action="store_true",
        help="Save best checkpoint by val rollout forecast ADE (px); add pixel L1 to loss",
    )
    parser.add_argument(
        "--px-l1-weight",
        type=float,
        default=0.25,
        help="Weight for pixel L1 term when --optimize-forecast-ade (default 0.25)",
    )
    args = parser.parse_args()

    require_rules = args.model == "rule_features"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.model == "rule_features" and args.residual:
        out_dir = lstm_out_dir(args.dataset) / "lstm_rule_features_residual"
    elif args.model == "rule_features" and args.split == "temporal_all":
        out_dir = lstm_out_dir(args.dataset) / "lstm_rule_features_temporal"
    else:
        out_dir = lstm_out_dir(args.dataset) / f"lstm_{args.model}"
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
    px_l1_w = args.px_l1_weight if args.optimize_forecast_ade else 0.0
    history = {"train_loss": [], "val_loss": [], "val_forecast_ade_px": []}
    val_paths = split_info.get("val_paths", [])
    run_cfg = {
        "model": args.model,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "residual": args.residual,
        "scale": scale.tolist(),
        "num_players": None,
    }

    best_ade = float("inf")
    best_state = None
    best_epoch = 0

    print(
        f"Training {args.model} on {device} | residual={args.residual} | "
        f"optimize_forecast_ade={args.optimize_forecast_ade} | "
        f"train={len(train_loader)} val={len(val_loader)}"
    )
    for epoch in range(1, args.epochs + 1):
        sched_p = scheduled_sampling_prob(epoch) if args.scheduled_sampling else 0.0
        tr = train_epoch(
            model,
            args.model,
            train_loader,
            optimizer,
            device,
            scale_t,
            rule_w,
            scheduled_p=sched_p,
            residual=args.residual,
            pred_len=args.pred_len,
            px_l1_weight=px_l1_w,
        )
        va = eval_epoch(
            model,
            args.model,
            val_loader,
            device,
            scale_t,
            rule_w,
            residual=args.residual,
            pred_len=args.pred_len,
            px_l1_weight=px_l1_w,
        )
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        val_ade = float("nan")
        if args.optimize_forecast_ade and val_paths:
            run_cfg["num_players"] = num_players
            val_ade = forecast_ade_px_rollout(model, run_cfg, val_paths, device)
            history["val_forecast_ade_px"].append(val_ade)
            if val_ade == val_ade and val_ade < best_ade:
                best_ade = val_ade
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            msg = f"Epoch {epoch:3d}  train={tr:.6f}  val={va:.6f}"
            if val_ade == val_ade:
                msg += f"  val_forecast_ade={val_ade:.3f}px"
                if best_ade == best_ade:
                    msg += f"  best={best_ade:.3f}px@{best_epoch}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best checkpoint from epoch {best_epoch} (val forecast ADE {best_ade:.3f} px)")

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
        "scheduled_sampling": args.scheduled_sampling,
        "residual": args.residual,
        "optimize_forecast_ade": args.optimize_forecast_ade,
        "px_l1_weight": px_l1_w,
        "best_val_forecast_ade_px": best_ade if best_ade == best_ade else None,
        "best_epoch": best_epoch if best_epoch else None,
    }

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")
    if best_ade == best_ade:
        print(f"Best val forecast ADE: {best_ade:.3f} px (epoch {best_epoch})")

    run_post_eval = args.post_eval or (
        args.model == "rule_features"
        and (args.split == "temporal_all" or args.residual)
    )
    if run_post_eval:
        eval_script = ROOT / "scripts" / "eval_lstm_ablations.py"
        extra = ["--a1-residual-checkpoint", str(ckpt_path)] if args.residual else []
        a1_flag = [] if args.residual else ["--a1-checkpoint", str(ckpt_path)]
        cmd = [
            sys.executable,
            str(eval_script),
            "--dataset",
            args.dataset,
            "--all-seeds",
            "--skip-attribution",
            *a1_flag,
            *extra,
        ]
        print("Running post-train multi-seed eval:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        summary_path = lstm_out_dir(args.dataset) / "lstm_ablation_summary.json"
        variant_key = "A1_residual" if args.residual else "A1_temporal_all"
        entry = {
            "variant": variant_key,
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "scheduled_sampling": args.scheduled_sampling,
            "residual": args.residual,
            "optimize_forecast_ade": args.optimize_forecast_ade,
            "best_val_forecast_ade_px": best_ade if best_ade == best_ade else None,
            "train_config": str(out_dir / "train_config.json"),
        }
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}
        if "runs" not in existing:
            if "results" in existing:
                existing = {
                    "runs": {"held_out_single_seed": existing},
                }
            else:
                existing = {"runs": {}}
        existing["runs"][variant_key] = entry
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"Appended {variant_key} to {summary_path}")



if __name__ == "__main__":
    main()
