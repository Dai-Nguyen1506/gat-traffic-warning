"""Data loading, deduplicated graph construction, and chronological regression splits for PeMS."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["Flow", "Occupancy", "Speed"]
TIME_FEATURE_NAMES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]

# Calendar start day-of-week (0=Mon .. 6=Sun): PEMS08 starts Fri 2016-07-01 (4), PEMS04 starts Mon 2018-01-01 (0)
DATASET_START_DOW = {
    "PEMS08": 4,
    "PEMS04": 0,
}


def load_clean_pems(
    dataset_name: str,
    clean_dir: str | Path = "datasets/cleaned",
    raw_dir: str | Path = "datasets/data",
    out_dir: str | Path = "outputs",
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load cleaned PeMS tensor (T, N, 3), deduplicated directed edges, and bottleneck table."""
    clean_path = Path(clean_dir) / f"{dataset_name}_clean.npz"
    csv_path = Path(raw_dir) / dataset_name / f"{dataset_name}.csv"
    bn_path = Path(out_dir) / f"{dataset_name}_bottlenecks.csv"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {clean_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing edge list CSV: {csv_path}")
    if not bn_path.exists():
        raise FileNotFoundError(f"Missing bottlenecks CSV: {bn_path}")

    npz_obj = np.load(clean_path)
    key = "data" if "data" in npz_obj.files else npz_obj.files[0]
    data = npz_obj[key].astype(np.float32)

    edges_df = pd.read_csv(csv_path)
    edges_df.columns = [c.strip().lower() for c in edges_df.columns]
    edges_df["from"] = edges_df["from"].astype(int)
    edges_df["to"] = edges_df["to"].astype(int)
    edges_df["cost"] = edges_df["cost"].astype(float)
    # Deduplicate directed edges consistently with Notebook 03 (277 unique for PEMS08)
    edges_df = (
        edges_df.sort_values("cost", ascending=True)
        .drop_duplicates(subset=["from", "to"], keep="first")
        .reset_index(drop=True)
    )

    bn_df = pd.read_csv(bn_path)
    bn_top = bn_df[bn_df["is_bottleneck"]].sort_values("congestion_freq", ascending=False)
    bottleneck_nodes = bn_top["node_id"].to_numpy(dtype=np.int32)

    num_nodes = data.shape[1]
    bottleneck_mask = np.zeros(num_nodes, dtype=bool)
    bottleneck_mask[bottleneck_nodes] = True

    return data, edges_df, bn_df, bottleneck_nodes, bottleneck_mask


def build_edge_index(
    edges_df: pd.DataFrame,
    num_nodes: int,
    add_self_loops: bool = True,
    bidirectional: bool = False,
) -> np.ndarray:
    """Return deduplicated COO edge_index (2, E) int32 with src=edge_index[0], dst=edge_index[1]."""
    src = edges_df["from"].to_numpy(dtype=np.int32)
    dst = edges_df["to"].to_numpy(dtype=np.int32)
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]

    if bidirectional:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])

    if add_self_loops:
        loops = np.arange(num_nodes, dtype=np.int32)
        src = np.concatenate([src, loops])
        dst = np.concatenate([dst, loops])

    edge_index = np.stack([src, dst], axis=0)
    keys = edge_index[0].astype(np.int64) * num_nodes + edge_index[1].astype(np.int64)
    _, unique_idx = np.unique(keys, return_index=True)
    unique_idx.sort()
    return edge_index[:, unique_idx].astype(np.int32)


def apply_congestion_rule(
    speed: np.ndarray,
    occ: np.ndarray,
    speed_th: float = 40.0,
    occ_th: float = 0.15,
) -> np.ndarray:
    """Decoupled physical congestion criterion finalized in Notebook 03."""
    return (speed < speed_th) & (occ >= occ_th)


def build_time_features(
    num_steps: int,
    steps_per_day: int = 288,
    start_dow: int = 4,
) -> np.ndarray:
    """Create cyclical time-of-day and day-of-week features of shape (T, 4)."""
    t = np.arange(num_steps, dtype=np.float32)
    tod = (t % steps_per_day) / float(steps_per_day)
    dow = ((t // steps_per_day) + start_dow) % 7.0 / 7.0

    feats = np.stack(
        [
            np.sin(2.0 * np.pi * tod),
            np.cos(2.0 * np.pi * tod),
            np.sin(2.0 * np.pi * dow),
            np.cos(2.0 * np.pi * dow),
        ],
        axis=-1,
    )
    return feats.astype(np.float32)


def create_regression_splits(
    data: np.ndarray,
    bottleneck_nodes: np.ndarray,
    history_steps: int = 8,
    lead_steps: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    start_dow: int = 4,
) -> Dict[str, object]:
    """
    Create strictly chronological 70/15/15 spatio-temporal regression splits with zero leakage.

    Inputs X have shape (M, N, H, 7): standardized [Flow, Occ, Speed] + [tod_sin, tod_cos, dow_sin, dow_cos].
    Targets y have shape (M, N, 3): standardized [Flow, Occ, Speed] at t + lead_steps.
    """
    T, N, C = data.shape
    t_train_end = int(T * train_ratio)
    t_val_end = int(T * (train_ratio + val_ratio))

    scaler = StandardScaler()
    scaler.fit(data[:t_train_end].reshape(-1, C))
    scaled = scaler.transform(data.reshape(-1, C)).reshape(T, N, C).astype(np.float32)

    time_feats = build_time_features(T, steps_per_day=288, start_dow=start_dow)  # (T, 4)
    time_broad = np.broadcast_to(time_feats[:, None, :], (T, N, 4))
    full_feats = np.concatenate([scaled, time_broad], axis=-1).astype(np.float32)  # (T, N, 7)

    def _make_windows(start_bound: int, end_bound: int):
        first_t = start_bound + history_steps - 1
        last_t = end_bound - lead_steps
        ts = np.arange(first_t, last_t, dtype=np.int32)
        X = np.stack(
            [full_feats[ts - history_steps + 1 + h] for h in range(history_steps)],
            axis=2,
        ).astype(np.float32)  # (M, N, H, 7)
        y_std = scaled[ts + lead_steps].astype(np.float32)  # (M, N, 3)
        y_raw = data[ts + lead_steps].astype(np.float32)  # (M, N, 3)
        cur_raw = data[ts].astype(np.float32)  # (M, N, 3)
        return X, y_std, y_raw, cur_raw, ts

    X_train, y_train, y_raw_train, cur_raw_train, t_train = _make_windows(0, t_train_end)
    X_val, y_val, y_raw_val, cur_raw_val, t_val = _make_windows(t_train_end, t_val_end)
    X_test, y_test, y_raw_test, cur_raw_test, t_test = _make_windows(t_val_end, T)

    bottleneck_mask = np.zeros(N, dtype=bool)
    bottleneck_mask[bottleneck_nodes] = True

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_raw_train": y_raw_train,
        "cur_raw_train": cur_raw_train,
        "t_train": t_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_raw_val": y_raw_val,
        "cur_raw_val": cur_raw_val,
        "t_val": t_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_raw_test": y_raw_test,
        "cur_raw_test": cur_raw_test,
        "t_test": t_test,
        "scaler": scaler,
        "bottleneck_mask": bottleneck_mask,
        "bottleneck_nodes": bottleneck_nodes,
        "history_steps": history_steps,
        "lead_steps": lead_steps,
        "t_train_end": t_train_end,
        "t_val_end": t_val_end,
    }


def inverse_scale_targets(y_std: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform standardized targets (..., 3) back to physical [Flow, Occupancy, Speed]."""
    orig_shape = y_std.shape
    raw = scaler.inverse_transform(y_std.reshape(-1, orig_shape[-1])).reshape(orig_shape)
    # Enforce physical non-negativity (Flow >= 0, Occ in [0, 1], Speed >= 0)
    raw = raw.astype(np.float32)
    raw[..., 0] = np.clip(raw[..., 0], 0.0, None)
    raw[..., 1] = np.clip(raw[..., 1], 0.0, 1.0)
    raw[..., 2] = np.clip(raw[..., 2], 0.0, None)
    return raw
