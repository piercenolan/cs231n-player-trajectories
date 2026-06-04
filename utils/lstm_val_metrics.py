"""Validation forecast ADE (pixel space) for training checkpoints."""

import numpy as np
import torch

from utils.lstm_dataset import load_tensor_file
from utils.lstm_predict import rollout_positions


@torch.no_grad()
def forecast_ade_px_rollout(model, cfg, tensor_paths, device, max_paths=None):
    """Mean forecast-horizon ADE in pixels (rollout, matches eval_lstm_ablations)."""
    model.eval()
    obs_len = cfg["obs_len"]
    paths = list(tensor_paths)
    if max_paths is not None:
        paths = paths[:max_paths]
    all_errs = []
    for tp in paths:
        seq = load_tensor_file(tp)
        scale = np.asarray(seq["scale"], dtype=np.float32)
        cfg_run = {**cfg, "scale": scale.tolist()}
        pred_pos, _ = rollout_positions(model, seq, cfg_run, device=device)
        gt = seq["positions"]
        vis = seq["visibility"]
        for t in range(obs_len, pred_pos.shape[0]):
            for p in range(pred_pos.shape[1]):
                if not vis[t, p]:
                    continue
                all_errs.append(
                    float(
                        np.hypot(
                            pred_pos[t, p, 0] - gt[t, p, 0],
                            pred_pos[t, p, 1] - gt[t, p, 1],
                        )
                    )
                )
    return float(np.mean(all_errs)) if all_errs else float("nan")
