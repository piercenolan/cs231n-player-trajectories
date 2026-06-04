"""LSTM forecaster for fixed-slot player trajectories."""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    Encode past frames of P players (flattened to P*2) and predict next pred_len frames.
    """

    def __init__(
        self,
        num_players=10,
        hidden_dim=128,
        num_layers=2,
        pred_len=4,
        dropout=0.1,
    ):
        super().__init__()
        self.num_players = num_players
        self.pred_len = pred_len
        self.input_dim = num_players * 2
        self.output_dim = pred_len * num_players * 2

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        """
        x: (batch, obs_len, P, 2) or (batch, obs_len, P*2)
        Returns: (batch, pred_len, P, 2)
        """
        if x.dim() == 4:
            b, t, p, c = x.shape
            x = x.reshape(b, t, p * c)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last)
        b = pred.shape[0]
        return pred.view(b, self.pred_len, self.num_players, 2)


def masked_mse(pred, target, mask):
    """
    pred, target: (batch, pred_len, P, 2)
    mask: bool (batch, pred_len, P)
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    diff = (pred - target) ** 2
    mask_f = mask.float().unsqueeze(-1).expand_as(diff)
    denom = mask_f.sum().clamp(min=1.0)
    return (diff * mask_f).sum() / denom
