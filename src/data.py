"""Zero-leakage PyTorch data pipeline for PeMS (60/20/20 split, 12->12 window, Train-only threshold & scaler)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.bottlenecks import (
    CSD_FEATURE_NAMES,
    CongestionThresholds,
    calibrate_congestion_thresholds,
    classify_traffic_phases,
    extract_precongestion_features,
    extract_time_features,
    load_cleaned_pems,
)

RAW_FEATURE_NAMES = ["Flow", "Occupancy", "Speed"]
TIME_FEATURE_NAMES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]

# Canonical 14-channel ordering: Raw(3) + Group 4 STID(4) + Group 1-3 Pre-Congestion Signals(7)
ALL_FEATURE_NAMES: List[str] = RAW_FEATURE_NAMES + TIME_FEATURE_NAMES + list(CSD_FEATURE_NAMES)

# Indices of continuous channels standardized using Train-only mean/std (sin/cos remain in [-1, 1])
CONTINUOUS_CHANNEL_INDICES: List[int] = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13]

# Named feature groups aligned with NB03 (RQ1)
FEATURE_GROUPS: Dict[str, List[int]] = OrderedDict(
    [
        ("Raw (3 biến gốc)", [0, 1, 2]),
        ("Nhóm 4 — STID (Chu kỳ thời gian)", [3, 4, 5, 6]),
        ("Nhóm 1 — Dao động khử dốc", [7, 8, 9]),
        ("Nhóm 2 — Liên kết đa biến", [10, 11]),
        ("Nhóm 3 — Độ dốc có dấu", [12, 13]),
    ]
)

# 6 controlled feature-set configurations on the frozen ST-GAT (LSTM -> GATv2) architecture
FEATURE_CONFIGS: Dict[str, List[int]] = OrderedDict(
    [
        ("Raw (3)", [0, 1, 2]),
        ("Base (7)", [0, 1, 2, 3, 4, 5, 6]),
        ("Base + Nhóm 1 (10)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("Base + Nhóm 2 (9)", [0, 1, 2, 3, 4, 5, 6, 10, 11]),
        ("Base + Nhóm 3 (9)", [0, 1, 2, 3, 4, 5, 6, 12, 13]),
        ("Full (14)", list(range(14))),
    ]
)

# Calendar start day-of-week (0=Mon .. 6=Sun): PEMS08 starts Fri 2016-07-01 (4), PEMS04 starts Mon 2018-01-01 (0)
DATASET_START_DOW: Dict[str, int] = {
    "PEMS08": 4,
    "PEMS04": 0,
}

# Edge direction types for GATv2
EDGE_TYPE_SELF = 0
EDGE_TYPE_IN = 1   # Upstream -> Downstream (original directed road link u -> v)
EDGE_TYPE_OUT = 2  # Downstream -> Upstream (reverse spillback link v -> u)


def build_directed_edge_index(
    edges_df: pd.DataFrame,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build deduplicated bidirectional edge_index (2, E) with explicit direction types:
      - 0 (Self): i -> i
      - 1 (In):   u -> v from original directed edges_df (upstream inflow)
      - 2 (Out):  v -> u reverse edges not already in original directed edges_df (downstream spillback).
    """
    orig_pairs = set()
    for row in edges_df[["from", "to"]].itertuples(index=False):
        u, v = int(row[0]), int(row[1])
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            orig_pairs.add((u, v))

    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []

    # 1. Self-loops
    for i in range(num_nodes):
        src_list.append(i)
        dst_list.append(i)
        type_list.append(EDGE_TYPE_SELF)

    # 2. Upstream -> Downstream (In)
    for u, v in sorted(orig_pairs):
        src_list.append(u)
        dst_list.append(v)
        type_list.append(EDGE_TYPE_IN)

    # 3. Downstream -> Upstream (Out / spillback), skipping mutual pairs already added as In
    n_out = 0
    for u, v in sorted(orig_pairs):
        if (v, u) not in orig_pairs:
            src_list.append(v)
            dst_list.append(u)
            type_list.append(EDGE_TYPE_OUT)
            n_out += 1

    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    edge_type = np.asarray(type_list, dtype=np.int64)
    edge_meta = {
        "n_nodes": int(num_nodes),
        "n_edges_total": int(len(type_list)),
        "n_self": int(num_nodes),
        "n_in": int(len(orig_pairs)),
        "n_out": int(n_out),
    }
    return edge_index, edge_type, edge_meta


def apply_thresholds(
    speed: np.ndarray,
    occ: np.ndarray,
    thresholds: CongestionThresholds,
) -> np.ndarray:
    """Apply Train-calibrated physical congestion rule: (Speed < v_cong) & (Occ >= o_sat)."""
    return (speed < thresholds.v_cong) & (occ >= thresholds.o_sat)


def inverse_scale_targets(
    y_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    clip_physical: bool = True,
) -> np.ndarray:
    """Inverse-transform standardized predictions (..., 3) back to physical [Flow, Occupancy, Speed]."""
    raw = (y_std * target_std + target_mean).astype(np.float32)
    if clip_physical:
        raw[..., 0] = np.clip(raw[..., 0], 0.0, None)
        raw[..., 1] = np.clip(raw[..., 1], 0.0, 1.0)
        raw[..., 2] = np.clip(raw[..., 2], 0.0, None)
    return raw


def prepare_pems_bundle(
    dataset_name: str = "PEMS08",
    clean_dir: str | Path = "datasets/cleaned",
    raw_dir: str | Path = "datasets/data",
    history_steps: int = 12,
    horizon_steps: int = 12,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    bottleneck_min_freq: float = 0.01,
) -> Dict[str, object]:
    """
    Prepare zero-leakage chronological 60/20/20 PeMS bundle for 12 -> 12 multi-step forecasting:
      1. Calibrate congestion thresholds (v_cong, o_sat) STRICTLY on Train (60%), then apply to Val & Test.
      2. Identify bottleneck stations (congestion_freq >= 1%) STRICTLY on Train (60%).
      3. Standardize continuous channels STRICTLY using Train (60%) mean/std.
      4. Enforce causal window margin (2 * H - 2 = 22 steps) inside each split so zero past data crosses split boundaries.
    """
    data, edges_df = load_cleaned_pems(dataset_name, clean_dir=clean_dir, raw_dir=raw_dir)
    T, N, _ = data.shape
    t_train_end = int(T * train_ratio)
    t_val_end = int(T * (train_ratio + val_ratio))

    # 1. Train-only congestion threshold calibration (zero leakage into Val/Test)
    train_thresholds = calibrate_congestion_thresholds(
        data[:t_train_end, :, 0],
        data[:t_train_end, :, 1],
        data[:t_train_end, :, 2],
    )
    _, is_congested, is_synchronized = classify_traffic_phases(
        data[:, :, 2], data[:, :, 1], train_thresholds
    )

    # Train-only bottleneck mask (stations with >= 1% congestion in Train split)
    train_cong_freq = is_congested[:t_train_end].mean(axis=0)
    bottleneck_mask = train_cong_freq >= bottleneck_min_freq
    bottleneck_nodes = np.where(bottleneck_mask)[0].astype(np.int64)
    bottleneck_nodes = bottleneck_nodes[np.argsort(-train_cong_freq[bottleneck_nodes])]

    # 2. Extract all 14 local features: Raw(3) + STID(4) + Pre-congestion CSD(7)
    start_dow = DATASET_START_DOW.get(dataset_name, 0)
    time4 = extract_time_features(T, N, start_dow=start_dow)
    csd7 = extract_precongestion_features(data, window=history_steps)
    feats_raw = np.concatenate([data, time4, csd7], axis=-1).astype(np.float32)  # (T, N, 14)

    # 3. Train-only standardization on continuous channels (skip initial padded csd steps < history_steps - 1)
    fit_slice = feats_raw[history_steps - 1 : t_train_end]
    feat_mean = np.zeros(14, dtype=np.float32)
    feat_std = np.ones(14, dtype=np.float32)
    for ch in CONTINUOUS_CHANNEL_INDICES:
        vals = fit_slice[:, :, ch]
        feat_mean[ch] = float(vals.mean())
        feat_std[ch] = float(max(vals.std(), 1e-6))

    feats_scaled = ((feats_raw - feat_mean) / feat_std).astype(np.float32)
    target_mean = feat_mean[:3].copy()
    target_std = feat_std[:3].copy()
    targets_scaled = feats_scaled[:, :, :3].copy()

    # 4. Strictly isolated anchor indices per split (earliest csd lookback = t - 2*H + 2 stays inside split)
    margin = 2 * history_steps - 2
    t_train = np.arange(margin, t_train_end - horizon_steps, dtype=np.int64)
    t_val = np.arange(t_train_end + margin, t_val_end - horizon_steps, dtype=np.int64)
    t_test = np.arange(t_val_end + margin, T - horizon_steps, dtype=np.int64)

    edge_index, edge_type, edge_meta = build_directed_edge_index(edges_df, num_nodes=N)

    return {
        "dataset_name": dataset_name,
        "data_raw": data,
        "edges_df": edges_df,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_meta": edge_meta,
        "feats_raw": feats_raw,
        "feats_scaled": feats_scaled,
        "targets_scaled": targets_scaled,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_thresholds": train_thresholds,
        "is_congested": is_congested,
        "is_synchronized": is_synchronized,
        "train_cong_freq": train_cong_freq,
        "bottleneck_mask": bottleneck_mask,
        "bottleneck_nodes": bottleneck_nodes,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "t_train_end": t_train_end,
        "t_val_end": t_val_end,
        "t_train": t_train,
        "t_val": t_val,
        "t_test": t_test,
    }


class PeMSWindowDataset(Dataset):
    """Memory-efficient PyTorch Dataset slicing (H, N, F) -> (N, P, 3) windows on the fly."""

    def __init__(
        self,
        feats_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        anchors: np.ndarray,
        feature_indices: Sequence[int],
        history_steps: int = 12,
        horizon_steps: int = 12,
    ) -> None:
        self.feats = torch.from_numpy(
            np.ascontiguousarray(feats_scaled[:, :, list(feature_indices)], dtype=np.float32)
        )
        self.targets = torch.from_numpy(np.ascontiguousarray(targets_scaled, dtype=np.float32))
        self.anchors = anchors.astype(np.int64)
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.anchors[idx])
        # x: (H, N, F), y: (N, P, 3)
        x = self.feats[t - self.history_steps + 1 : t + 1]
        y = self.targets[t + 1 : t + self.horizon_steps + 1].permute(1, 0, 2)
        return x, y
