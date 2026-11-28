"""PyTorch explainability for ST-GAT (LSTM -> GATv2): Permutation Feature Importance & Edge Attention."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.data import (
    ALL_FEATURE_NAMES,
    EDGE_TYPE_IN,
    EDGE_TYPE_OUT,
    EDGE_TYPE_SELF,
    FEATURE_GROUPS,
    apply_thresholds,
    inverse_scale_targets,
)
from src.models import STGATv2
from src.train import iter_vectorized_batches


def _eval_batch_mae(
    model: STGATv2,
    X_all: np.ndarray,
    y_true_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    true_cong_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Fast batched MAE computation for permutation importance."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            xb = torch.from_numpy(X_all[i : i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    y_pred_std = np.concatenate(preds, axis=0)
    y_true = inverse_scale_targets(y_true_std, target_mean, target_std, clip_physical=False)
    y_pred = inverse_scale_targets(y_pred_std, target_mean, target_std, clip_physical=True)

    abs_err = np.abs(y_pred - y_true)
    return {
        "std_mae": float(np.abs(y_pred_std - y_true_std).mean()),
        "mae_speed": float(abs_err[..., 2].mean()),
        "mae_occ": float(abs_err[..., 1].mean()),
        "mae_flow": float(abs_err[..., 0].mean()),
        "mae_speed_cong": float(abs_err[..., 2][true_cong_mask].mean()) if true_cong_mask.any() else float("nan"),
    }


def compute_permutation_importance(
    model: STGATv2,
    bundle: Dict[str, object],
    feature_indices: Sequence[int] = tuple(range(14)),
    n_repeats: int = 5,
    max_samples: int = 1024,
    batch_size: int = 128,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Compute Test-set Permutation Feature Importance (increase in MAE when permuting each feature or group
    across time samples) on the trained ST-GAT (Full 14) model.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    H = int(bundle["history_steps"])
    P = int(bundle["horizon_steps"])
    anchors = bundle["t_test"]
    if len(anchors) > max_samples:
        pick = np.sort(rng.choice(len(anchors), size=max_samples, replace=False))
        anchors = anchors[pick]

    feats_t = torch.from_numpy(
        np.ascontiguousarray(bundle["feats_scaled"][:, :, list(feature_indices)], dtype=np.float32)
    )
    targets_t = torch.from_numpy(np.ascontiguousarray(bundle["targets_scaled"], dtype=np.float32))
    h_offs = np.arange(-H + 1, 1, dtype=np.int64)
    p_offs = np.arange(1, P + 1, dtype=np.int64)
    X_base = feats_t[torch.from_numpy(anchors[:, None] + h_offs[None, :])].numpy()  # (M, H, N, F)
    y_std = targets_t[torch.from_numpy(anchors[:, None] + p_offs[None, :])].permute(0, 2, 1, 3).numpy()  # (M, N, P, 3)

    t_mean, t_std = bundle["target_mean"], bundle["target_std"]
    y_raw = inverse_scale_targets(y_std, t_mean, t_std, clip_physical=False)
    true_cong = apply_thresholds(y_raw[..., 2], y_raw[..., 1], bundle["train_thresholds"])

    base_m = _eval_batch_mae(model, X_base, y_std, t_mean, t_std, true_cong, device, batch_size)
    M = len(X_base)

    # 1. Per-feature Permutation Importance (all 14 features)
    feat_to_group = {}
    for gname, idxs in FEATURE_GROUPS.items():
        for idx in idxs:
            feat_to_group[idx] = gname

    feat_rows = []
    for pos, ch in enumerate(feature_indices):
        d_std, d_spd, d_occ, d_flw, d_cong = [], [], [], [], []
        for _ in range(n_repeats):
            perm = rng.permutation(M)
            X_perm = X_base.copy()
            X_perm[:, :, :, pos] = X_base[perm, :, :, pos]
            pm = _eval_batch_mae(model, X_perm, y_std, t_mean, t_std, true_cong, device, batch_size)
            d_std.append(pm["std_mae"] - base_m["std_mae"])
            d_spd.append(pm["mae_speed"] - base_m["mae_speed"])
            d_occ.append(pm["mae_occ"] - base_m["mae_occ"])
            d_flw.append(pm["mae_flow"] - base_m["mae_flow"])
            d_cong.append(pm["mae_speed_cong"] - base_m["mae_speed_cong"])

        feat_rows.append(
            {
                "channel": int(ch),
                "feature": ALL_FEATURE_NAMES[ch],
                "group": feat_to_group[ch],
                "delta_std_mae": float(np.mean(d_std)),
                "delta_std_mae_sd": float(np.std(d_std)),
                "delta_mae_speed": float(np.mean(d_spd)),
                "delta_mae_speed_sd": float(np.std(d_spd)),
                "delta_mae_occ": float(np.mean(d_occ)),
                "delta_mae_flow": float(np.mean(d_flw)),
                "delta_mae_speed_cong": float(np.mean(d_cong)),
            }
        )
    feature_df = (
        pd.DataFrame(feat_rows)
        .sort_values("delta_std_mae", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Per-group Permutation Importance (permute all channels of a group jointly)
    pos_map = {int(ch): pos for pos, ch in enumerate(feature_indices)}
    group_rows = []
    for gname, chs in FEATURE_GROUPS.items():
        positions = [pos_map[c] for c in chs if c in pos_map]
        if not positions:
            continue
        d_std, d_spd, d_occ, d_flw, d_cong = [], [], [], [], []
        for _ in range(n_repeats):
            perm = rng.permutation(M)
            X_perm = X_base.copy()
            for p_i in positions:
                X_perm[:, :, :, p_i] = X_base[perm, :, :, p_i]
            pm = _eval_batch_mae(model, X_perm, y_std, t_mean, t_std, true_cong, device, batch_size)
            d_std.append(pm["std_mae"] - base_m["std_mae"])
            d_spd.append(pm["mae_speed"] - base_m["mae_speed"])
            d_occ.append(pm["mae_occ"] - base_m["mae_occ"])
            d_flw.append(pm["mae_flow"] - base_m["mae_flow"])
            d_cong.append(pm["mae_speed_cong"] - base_m["mae_speed_cong"])

        group_rows.append(
            {
                "group": gname,
                "n_features": len(positions),
                "delta_std_mae": float(np.mean(d_std)),
                "delta_std_mae_sd": float(np.std(d_std)),
                "delta_mae_speed": float(np.mean(d_spd)),
                "delta_mae_speed_sd": float(np.std(d_spd)),
                "delta_mae_occ": float(np.mean(d_occ)),
                "delta_mae_flow": float(np.mean(d_flw)),
                "delta_mae_speed_cong": float(np.mean(d_cong)),
            }
        )
    group_df = (
        pd.DataFrame(group_rows)
        .sort_values("delta_std_mae", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "baseline_metrics": base_m,
        "feature_df": feature_df,
        "group_df": group_df,
        "n_eval_samples": int(M),
    }


@torch.no_grad()
def extract_gatv2_edge_attention(
    model: STGATv2,
    bundle: Dict[str, object],
    feature_indices: Sequence[int] = tuple(range(14)),
    target_horizon_step: int = 3,
    top_node: int | None = None,
    batch_size: int = 128,
) -> Dict[str, object]:
    """
    Extract dynamic GATv2 attention weights alpha_ij on Test split, separated by
    Self vs. Upstream (In) vs. Downstream spillback (Out), contrasting Free-flow vs. Pre-Congestion
    (true congestion at t + target_horizon_step * 5 min under Train-calibrated thresholds).
    """
    model.eval()
    device = next(model.parameters()).device
    H = int(bundle["history_steps"])
    P = int(bundle["horizon_steps"])
    feats_t = torch.from_numpy(
        np.ascontiguousarray(bundle["feats_scaled"][:, :, list(feature_indices)], dtype=np.float32)
    )
    targets_t = torch.from_numpy(np.ascontiguousarray(bundle["targets_scaled"], dtype=np.float32))

    attn_list, y_list = [], []
    for xb, yb in iter_vectorized_batches(
        feats_t, targets_t, bundle["t_test"], H, P, batch_size=batch_size, shuffle=False
    ):
        _, alpha = model.forward_with_attention(xb.to(device))
        attn_list.append(alpha.cpu().numpy())
        y_list.append(yb.numpy())
    all_attn = np.concatenate(attn_list, axis=0)  # (M_test, E)
    y_std = np.concatenate(y_list, axis=0)
    y_raw = inverse_scale_targets(y_std, bundle["target_mean"], bundle["target_std"], clip_physical=False)

    h_idx = target_horizon_step - 1
    true_cong = apply_thresholds(
        y_raw[:, :, h_idx, 2], y_raw[:, :, h_idx, 1], bundle["train_thresholds"]
    )  # (M_test, N)

    edge_index = bundle["edge_index"]
    edge_type = bundle["edge_type"]
    src, dst = edge_index[0], edge_index[1]
    type_names = {EDGE_TYPE_SELF: "Self", EDGE_TYPE_IN: "In (Thượng nguồn)", EDGE_TYPE_OUT: "Out (Hạ nguồn)"}

    bn_nodes = bundle["bottleneck_nodes"]
    if top_node is None:
        top_node = int(bn_nodes[0])

    # 1. Detailed edges feeding into primary bottleneck node
    in_top = np.where(dst == top_node)[0]
    cm_top = true_cong[:, top_node]
    nm_top = ~cm_top
    top_rows = []
    for e_idx in in_top:
        u, v, et = int(src[e_idx]), int(dst[e_idx]), int(edge_type[e_idx])
        a_all = float(all_attn[:, e_idx].mean())
        a_cong = float(all_attn[cm_top, e_idx].mean()) if cm_top.any() else a_all
        a_norm = float(all_attn[nm_top, e_idx].mean()) if nm_top.any() else a_all
        top_rows.append(
            {
                "src": u,
                "dst": v,
                "direction": type_names[et],
                "edge_type": et,
                "alpha_overall": a_all,
                "alpha_normal": a_norm,
                "alpha_pre_cong": a_cong,
                "alpha_shift": a_cong - a_norm,
            }
        )
    top_edges_df = (
        pd.DataFrame(top_rows)
        .sort_values("alpha_pre_cong", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Aggregated Self / In / Out attention across all Train-calibrated bottleneck stations
    bn_rows = []
    for bn in bn_nodes:
        bn_i = int(bn)
        cm = true_cong[:, bn_i]
        nm = ~cm
        if not (cm.any() and nm.any()):
            continue
        e_to_bn = np.where(dst == bn_i)[0]
        self_e = e_to_bn[edge_type[e_to_bn] == EDGE_TYPE_SELF]
        in_e = e_to_bn[edge_type[e_to_bn] == EDGE_TYPE_IN]
        out_e = e_to_bn[edge_type[e_to_bn] == EDGE_TYPE_OUT]
        nbr_e = e_to_bn[edge_type[e_to_bn] != EDGE_TYPE_SELF]

        def _sm(idxs, mask):
            return float(all_attn[mask][:, idxs].sum(axis=1).mean()) if len(idxs) else 0.0

        bn_rows.append(
            {
                "node_id": bn_i,
                "n_in": int(len(in_e)),
                "n_out": int(len(out_e)),
                "self_normal": _sm(self_e, nm),
                "self_pre_cong": _sm(self_e, cm),
                "in_normal": _sm(in_e, nm),
                "in_pre_cong": _sm(in_e, cm),
                "out_normal": _sm(out_e, nm),
                "out_pre_cong": _sm(out_e, cm),
                "nbr_normal": _sm(nbr_e, nm),
                "nbr_pre_cong": _sm(nbr_e, cm),
                "nbr_shift": _sm(nbr_e, cm) - _sm(nbr_e, nm),
            }
        )
    bn_summary_df = pd.DataFrame(bn_rows).sort_values("nbr_shift", ascending=False).reset_index(drop=True)

    return {
        "top_node": int(top_node),
        "top_edges_df": top_edges_df,
        "bn_summary_df": bn_summary_df,
        "summary_means": {
            "self_normal": float(bn_summary_df["self_normal"].mean()),
            "self_pre_cong": float(bn_summary_df["self_pre_cong"].mean()),
            "in_normal": float(bn_summary_df["in_normal"].mean()),
            "in_pre_cong": float(bn_summary_df["in_pre_cong"].mean()),
            "out_normal": float(bn_summary_df["out_normal"].mean()),
            "out_pre_cong": float(bn_summary_df["out_pre_cong"].mean()),
            "nbr_normal": float(bn_summary_df["nbr_normal"].mean()),
            "nbr_pre_cong": float(bn_summary_df["nbr_pre_cong"].mean()),
        },
    }
