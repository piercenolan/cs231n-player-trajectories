"""Graph-style social LSTM: relative neighbor offsets + temporal LSTM."""

import torch
import torch.nn as nn

from models.trajectory_lstm import masked_mse


class TrajectoryGraphLSTM(nn.Module):
    """
    Per timestep: encode each player with mean relative offset to visible neighbors,
    then LSTM over flattened multi-player state.
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
        self.node_in = 4  # x, y, mean_rel_x, mean_rel_y
        self.enc_dim = 32
        self.encoder = nn.Sequential(
            nn.Linear(self.node_in, self.enc_dim),
            nn.ReLU(),
            nn.Linear(self.enc_dim, self.enc_dim),
        )
        self.lstm = nn.LSTM(
            input_size=num_players * self.enc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, pred_len * num_players * 2)

    def _encode_frame(self, pos, vis_mask):
        """pos: (B, P, 2), vis_mask: (B, P) bool -> (B, P, enc_dim)."""
        b, p, _ = pos.shape
        rel_sum = torch.zeros(b, p, 2, device=pos.device, dtype=pos.dtype)
        counts = torch.zeros(b, p, 1, device=pos.device, dtype=pos.dtype)
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                m = vis_mask[:, i] & vis_mask[:, j]
                if not m.any():
                    continue
                diff = pos[:, j] - pos[:, i]
                rel_sum[:, i] += diff * m.unsqueeze(-1).float()
                counts[:, i] += m.unsqueeze(-1).float()
        rel_mean = rel_sum / counts.clamp(min=1.0)
        node_in = torch.cat([pos, rel_mean], dim=-1)
        return self.encoder(node_in)

    def forward(self, x, mask_x=None):
        """
        x: (B, T, P, 2)
        mask_x: (B, T, P) optional visibility for neighbor graph
        """
        if x.dim() != 4:
            raise ValueError("TrajectoryGraphLSTM expects (B, T, P, 2)")
        b, t, p, _ = x.shape
        if mask_x is None:
            mask_x = torch.ones(b, t, p, dtype=torch.bool, device=x.device)
        encoded = []
        for ti in range(t):
            encoded.append(self._encode_frame(x[:, ti], mask_x[:, ti]))
        seq = torch.stack(encoded, dim=1).reshape(b, t, p * self.enc_dim)
        out, _ = self.lstm(seq)
        pred = self.head(out[:, -1])
        return pred.view(b, self.pred_len, self.num_players, 2)


__all__ = ["TrajectoryGraphLSTM", "masked_mse"]
