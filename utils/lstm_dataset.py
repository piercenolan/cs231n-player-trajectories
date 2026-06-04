"""
PyTorch dataset for sliding-window trajectory forecasting from tensor JSON.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.datasets import lstm_tensor_paths, trajectory_tensor_path
from utils.seed_schedule import parse_seed_offset_sec


TRAIN_SEEDS = ("offset_10s", "offset_15s")
VAL_SEED = "offset_0s"


def load_tensor_file(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    positions = np.array(data["positions"], dtype=np.float32)
    visibility = np.array(data["visibility"], dtype=bool)
    meta = dict(data.get("meta") or {})
    seed_id = meta.get("seed_id")
    if not seed_id and "seeds" in path.parts:
        for part in path.parts:
            if parse_seed_offset_sec(part) is not None:
                seed_id = part
                break
    if not seed_id and parse_seed_offset_sec(path.parent.name) is not None:
        seed_id = path.parent.name
    scale = norm_stats_from_meta(meta)
    return {
        "positions": positions,
        "visibility": visibility,
        "frame_numbers": np.array(data["frame_numbers"], dtype=np.int32),
        "player_ids": data["player_ids"],
        "meta": meta,
        "scale": scale,
        "seed_id": seed_id or "root",
        "path": str(path),
    }


def norm_stats_from_meta(meta):
    w = float(meta.get("frame_width") or 1.0)
    h = float(meta.get("frame_height") or 1.0)
    return np.array([w, h], dtype=np.float32)


def normalize_positions(positions, scale):
    return positions / scale.reshape(1, 1, 2)


def denormalize_positions(positions, scale):
    return positions * scale.reshape(1, 1, 2)


class TrajectoryWindowDataset(Dataset):
    """Sliding windows: obs_len past -> pred_len future, masked by visibility."""

    def __init__(
        self,
        tensor_paths,
        obs_len=8,
        pred_len=4,
        stride=1,
        normalize=True,
        scale=None,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.normalize = normalize
        self.samples = []
        self.sequences = []

        for tp in tensor_paths:
            seq = load_tensor_file(tp)
            pos = seq["positions"]
            vis = seq["visibility"]
            scale = norm_stats_from_meta(seq["meta"]) if normalize else np.ones(2, np.float32)
            if scale is None or (scale <= 0).any():
                scale = np.array([1.0, 1.0], dtype=np.float32)
            seq["scale"] = scale
            if normalize:
                pos = normalize_positions(pos, scale)
            seq["positions_norm"] = pos
            self.sequences.append(seq)

            T = pos.shape[0]
            win = obs_len + pred_len
            for start in range(0, T - win + 1, stride):
                self.samples.append((len(self.sequences) - 1, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_i, start = self.samples[idx]
        seq = self.sequences[seq_i]
        pos = seq["positions_norm"]
        vis = seq["visibility"]
        t0 = start
        t_obs = t0 + self.obs_len
        t_end = t_obs + self.pred_len

        x = pos[t0:t_obs]
        y = pos[t_obs:t_end]
        mask_x = vis[t0:t_obs]
        mask_y = vis[t_obs:t_end]

        return {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
            "mask_x": torch.from_numpy(mask_x.copy()),
            "mask_y": torch.from_numpy(mask_y.copy()),
            "seed_id": seq["seed_id"],
            "start": start,
        }


def split_tensor_paths_by_seed(tensor_paths, val_seed=VAL_SEED):
    """Train on all seeds except val_seed; val on val_seed only."""
    train_paths = []
    val_paths = []
    for p in tensor_paths:
        seq = load_tensor_file(p)
        sid = seq["seed_id"]
        if sid == val_seed:
            val_paths.append(p)
        else:
            train_paths.append(p)
    return train_paths, val_paths


def resolve_tensor_paths(dataset, tensor_mode="multi"):
    if tensor_mode == "single":
        paths = lstm_tensor_paths(dataset, mode="single")
        if not paths:
            p = trajectory_tensor_path(dataset, seed_id="offset_0s")
            if p.exists():
                paths = [p]
        return paths
    return lstm_tensor_paths(dataset, mode="multi")


def build_dataloaders(
    dataset="sportsmot_example",
    tensor_mode="multi",
    obs_len=8,
    pred_len=4,
    stride=1,
    batch_size=16,
    val_seed=VAL_SEED,
    split="held_out_seed",
    num_workers=0,
):
    all_paths = resolve_tensor_paths(dataset, tensor_mode)
    if not all_paths:
        raise FileNotFoundError(
            f"No trajectory_tensors.json for {dataset}. Run:\n"
            f"  py scripts/export_lstm_tensors.py --dataset {dataset} --all-seeds --root"
        )

    if tensor_mode == "single":
        # Temporal split: first 80% windows train, last 20% val
        ds = TrajectoryWindowDataset(all_paths, obs_len, pred_len, stride)
        n = len(ds)
        split = max(1, int(n * 0.8))
        train_ds = torch.utils.data.Subset(ds, range(0, split))
        val_ds = torch.utils.data.Subset(ds, range(split, n))
        meta_scale = ds.sequences[0]["scale"]
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            meta_scale,
            {"train_paths": [str(all_paths[0])], "val_paths": [str(all_paths[0])], "split": "temporal"},
        )

    if split == "all_seeds":
        train_paths = list(all_paths)
        val_paths = list(all_paths)
        train_ds = TrajectoryWindowDataset(train_paths, obs_len, pred_len, stride)
        val_ds = train_ds
        meta_scale = train_ds.sequences[0]["scale"]
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            meta_scale,
            {
                "train_paths": [str(p) for p in train_paths],
                "val_paths": [str(p) for p in val_paths],
                "split": "all_seeds",
            },
        )

    if split == "temporal_all":
        ds = TrajectoryWindowDataset(all_paths, obs_len, pred_len, stride)
        n = len(ds)
        cut = max(1, int(n * 0.8))
        train_ds = torch.utils.data.Subset(ds, range(0, cut))
        val_ds = torch.utils.data.Subset(ds, range(cut, n))
        meta_scale = ds.sequences[0]["scale"]
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            meta_scale,
            {
                "train_paths": [str(p) for p in all_paths],
                "val_paths": [str(p) for p in all_paths],
                "split": "temporal_all",
                "num_train_windows": cut,
                "num_val_windows": n - cut,
            },
        )

    train_paths, val_paths = split_tensor_paths_by_seed(all_paths, val_seed=val_seed)
    if not train_paths:
        train_paths = [p for p in all_paths if load_tensor_file(p)["seed_id"] != val_seed]
    if not val_paths:
        val_paths = [p for p in all_paths if load_tensor_file(p)["seed_id"] == val_seed]
    if not val_paths:
        val_paths = [all_paths[0]]

    train_ds = TrajectoryWindowDataset(train_paths, obs_len, pred_len, stride)
    val_ds = TrajectoryWindowDataset(val_paths, obs_len, pred_len, stride)
    meta_scale = train_ds.sequences[0]["scale"] if train_ds.sequences else np.ones(2, np.float32)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        meta_scale,
        {
            "train_paths": [str(p) for p in train_paths],
            "val_paths": [str(p) for p in val_paths],
            "val_seed": val_seed,
            "split": "held_out_seed",
        },
    )
