"""Constant-velocity linear extrapolation for trajectory windows."""

import numpy as np
import torch


def linear_prediction_norm(x_obs, pred_len):
    """
    Extrapolate from normalized observation window.

    x_obs: (obs_len, P, 2) or (B, obs_len, P, 2)
    Returns: (pred_len, P, 2) or (B, pred_len, P, 2)
    """
    single = x_obs.ndim == 3
    x = np.asarray(x_obs, dtype=np.float32)
    if single:
        x = x[np.newaxis, ...]
    if x.shape[1] >= 2:
        vel = x[:, -1] - x[:, -2]
    else:
        vel = np.zeros_like(x[:, -1])
    preds = [x[:, -1] + vel * float(k) for k in range(1, pred_len + 1)]
    out = np.stack(preds, axis=1)
    return out[0] if single else out


def linear_prediction_norm_torch(x_obs, pred_len):
    """x_obs: (B, obs_len, P, 2) -> (B, pred_len, P, 2)"""
    if x_obs.shape[1] >= 2:
        vel = x_obs[:, -1] - x_obs[:, -2]
    else:
        vel = torch.zeros_like(x_obs[:, -1])
    preds = [x_obs[:, -1] + vel * float(k) for k in range(1, pred_len + 1)]
    return torch.stack(preds, dim=1)
