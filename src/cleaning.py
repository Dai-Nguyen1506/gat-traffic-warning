"""Data quality diagnostics and spatial linear regression imputation strictly following Chen et al. (TRR 2003)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_raw_pems(
    dataset_name: str,
    raw_dir: str | Path = "datasets/data",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load raw PeMS tensor (T, N, 3) [Flow, Occupancy, Speed] and deduplicated directed edges."""
    npz_path = Path(raw_dir) / dataset_name / f"{dataset_name}.npz"
    csv_path = Path(raw_dir) / dataset_name / f"{dataset_name}.csv"
    npz_obj = np.load(npz_path)
    key = "data" if "data" in npz_obj.files else npz_obj.files[0]
    data = npz_obj[key].astype(np.float32)

    edges_df = pd.read_csv(csv_path)
    edges_df.columns = [c.strip().lower() for c in edges_df.columns]
    edges_df["from"] = edges_df["from"].astype(int)
    edges_df["to"] = edges_df["to"].astype(int)
    edges_df["cost"] = edges_df["cost"].astype(float)
    num_nodes = data.shape[1]
    valid = (
        (edges_df["from"] >= 0)
        & (edges_df["from"] < num_nodes)
        & (edges_df["to"] >= 0)
        & (edges_df["to"] < num_nodes)
        & (edges_df["from"] != edges_df["to"])
    )
    edges_df = (
        edges_df[valid]
        .sort_values("cost", ascending=True)
        .drop_duplicates(subset=["from", "to"], keep="first")
        .reset_index(drop=True)
    )
    return data, edges_df


@dataclass
class ChenDiagnosticResult:
    daily_S1_zero_occ: np.ndarray      # (D, N) count of Occ == 0 per sensor-day
    daily_S2_desync: np.ndarray        # (D, N) count of Flow/Occ zero desync per sensor-day
    daily_S3_high_occ: np.ndarray      # (D, N) count of extreme high Occ (>= 0.70) per sensor-day
    daily_S4_entropy: np.ndarray       # (D, N) daily Shannon entropy (bits) of Occupancy
    p1_day: float
    p1_night: float
    p95_night_zero_dur: int
    n_clean_ref_nodes: int
    zero_valid_night_mask: np.ndarray  # (T, N) bool — valid empty road at night (preserved)
    zero_fault_mask: np.ndarray        # (T, N) bool — Type A (S1/S2) zero sensor dropout
    frozen_fault_mask: np.ndarray      # (T, N) bool — Type B (S4) non-zero frozen sensor streak
    fault_mask_3ch: np.ndarray         # (T, N, 3) bool — True means faulty and imputed


def run_chen2003_diagnostics(
    data: np.ndarray,
    steps_per_day: int = 288,
    frozen_min_steps: int = 12,
    p1_day_override: Optional[float] = None,
    p1_night_override: Optional[float] = None,
    p95_dur_override: Optional[int] = None,
) -> ChenDiagnosticResult:
    """
    Apply Chen et al. (TRR 2003) 4 daily diagnostic tests (S1..S4) plus step-level
    fault masking separating valid empty-road night zeros, Type A zero dropouts (S1/S2),
    and Type B non-zero frozen/flatlined sensor faults (S4 entropy).
    """
    T, N, _ = data.shape
    D = T // steps_per_day
    flow = data[:, :, 0]
    occ = data[:, :, 1]
    spd = data[:, :, 2]

    f_day = flow[: D * steps_per_day].reshape(D, steps_per_day, N)
    o_day = occ[: D * steps_per_day].reshape(D, steps_per_day, N)

    # Chen (2003) Daily Diagnostic Statistics (S1..S4)
    S1 = (o_day <= 0.0).sum(axis=1).astype(np.int32)
    S2 = (((o_day > 0.0) & (f_day <= 0.0)) | ((f_day > 0.0) & (o_day <= 0.0))).sum(axis=1).astype(np.int32)
    S3 = (o_day >= 0.70).sum(axis=1).astype(np.int32)
    S4 = np.zeros((D, N), dtype=np.float32)
    for d in range(D):
        for i in range(N):
            binned = np.round(o_day[d, :, i] * 200.0).astype(np.int32)
            _, cnts = np.unique(binned, return_counts=True)
            probs = cnts / float(steps_per_day)
            S4[d, i] = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # Empirical thresholds from zero-free clean sensor subset
    tod_hour = (np.arange(T) % steps_per_day) / 12.0
    is_daytime = (tod_hour >= 6.0) & (tod_hour <= 22.0)
    has_zero_node = ((flow <= 0.0) | (occ <= 0.0)).any(axis=0)
    clean_nodes = ~has_zero_node
    n_clean_ref = int(clean_nodes.sum())

    if p1_day_override is None or p1_night_override is None:
        ref_flow = flow[:, clean_nodes] if clean_nodes.any() else flow
        p1_day = float(np.percentile(ref_flow[is_daytime], 1))
        p1_night = float(np.percentile(ref_flow[~is_daytime], 1))
    else:
        p1_day, p1_night = float(p1_day_override), float(p1_night_override)

    is_zero_any = (flow <= 0.0) | (occ <= 0.0)
    desync = ((occ > 0.0) & (flow <= 0.0)) | ((flow > 0.0) & (occ <= 0.0))

    # Empirical P95 duration of low-flow purely nighttime zero runs
    night_low_runs: List[int] = []
    if p95_dur_override is None:
        for i in range(N):
            z = is_zero_any[:, i]
            if not z.any():
                continue
            pad = np.concatenate([[False], z, [False]])
            idx = np.where(pad[1:] != pad[:-1])[0]
            for a, b in zip(idx[0::2], idx[1::2]):
                if (~is_daytime[a:b]).all() and (~desync[a:b, i]).all():
                    adj_vals = []
                    if a > 0 and not z[a - 1]:
                        adj_vals.append(flow[a - 1, i])
                    if b < T and not z[b]:
                        adj_vals.append(flow[b, i])
                    if adj_vals and max(adj_vals) <= p1_night:
                        night_low_runs.append(int(b - a))
        p95_dur = int(np.percentile(night_low_runs, 95)) if night_low_runs else 6
    else:
        p95_dur = int(p95_dur_override)

    # Type A (S1/S2): Zero-value sensor dropouts vs valid empty-road night zeros
    zero_fault = np.zeros((T, N), dtype=bool)
    zero_valid_night = np.zeros((T, N), dtype=bool)
    for i in range(N):
        z = is_zero_any[:, i]
        if not z.any():
            continue
        pad = np.concatenate([[False], z, [False]])
        idx = np.where(pad[1:] != pad[:-1])[0]
        for a, b in zip(idx[0::2], idx[1::2]):
            ln = b - a
            adj_vals = []
            if a > 0 and not z[a - 1]:
                adj_vals.append(flow[a - 1, i])
            if b < T and not z[b]:
                adj_vals.append(flow[b, i])
            max_adj = max(adj_vals) if adj_vals else 0.0

            is_fault = (
                (ln > p95_dur)
                or bool(desync[a:b, i].any())
                or (bool(is_daytime[a:b].any()) and max_adj >= p1_night)
                or (max_adj >= p1_day)
            )
            if is_fault:
                zero_fault[a:b, i] = True
            else:
                zero_valid_night[a:b, i] = True

    # Type B (S4 Entropy): Non-zero frozen/flatlined sensor streaks (>= frozen_min_steps)
    frozen_fault = np.zeros((T, N), dtype=bool)
    spd_frozen = np.zeros((T, N), dtype=bool)
    for i in range(N):
        f_i, o_i, s_i = flow[:, i], occ[:, i], spd[:, i]
        same_fo = (f_i[1:] == f_i[:-1]) & (o_i[1:] == o_i[:-1]) & (f_i[1:] > 0.0) & (o_i[1:] > 0.0)
        pad = np.concatenate([[False], same_fo, [False]])
        idx = np.where(pad[1:] != pad[:-1])[0]
        for a, b in zip(idx[0::2], idx[1::2]):
            ln = (b - a) + 1
            if ln >= frozen_min_steps:
                frozen_fault[a : b + 1, i] = True
                if float(np.std(s_i[a : b + 1])) < 0.05:
                    spd_frozen[a : b + 1, i] = True

        # Speed flatlined simultaneously with Type A zero-dropout runs
        same_s = s_i[1:] == s_i[:-1]
        pad_s = np.concatenate([[False], same_s, [False]])
        idx_s = np.where(pad_s[1:] != pad_s[:-1])[0]
        for a, b in zip(idx_s[0::2], idx_s[1::2]):
            if (b - a) + 1 >= frozen_min_steps:
                overlap = zero_fault[a : b + 1, i]
                if overlap.any():
                    spd_frozen[a : b + 1, i] |= overlap

    fault_3ch = np.zeros((T, N, 3), dtype=bool)
    fault_3ch[:, :, 0] = zero_fault | frozen_fault
    fault_3ch[:, :, 1] = zero_fault | frozen_fault
    fault_3ch[:, :, 2] = spd_frozen

    return ChenDiagnosticResult(
        daily_S1_zero_occ=S1,
        daily_S2_desync=S2,
        daily_S3_high_occ=S3,
        daily_S4_entropy=S4,
        p1_day=p1_day,
        p1_night=p1_night,
        p95_night_zero_dur=p95_dur,
        n_clean_ref_nodes=n_clean_ref,
        zero_valid_night_mask=zero_valid_night,
        zero_fault_mask=zero_fault,
        frozen_fault_mask=frozen_fault,
        fault_mask_3ch=fault_3ch,
    )


def impute_chen2003_spatial_regression(
    data: np.ndarray,
    fault_mask_3ch: np.ndarray,
    edges_df: pd.DataFrame,
    top_k: int = 5,
    steps_per_day: int = 288,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Chen et al. (TRR 2003) pairwise spatial linear regression imputation:
    For each faulty sensor i, fit x_i(t) = a_ij * x_j(t) + b_ij against up to top_k
    correlated neighbors j (prioritizing 1-hop and 2-hop road-graph neighbors) strictly
    on jointly clean timestamps, and combine operational neighbor predictions weighted by R^2.
    """
    out = data.copy().astype(np.float32)
    T, N, C = out.shape
    adj = [set() for _ in range(N)]
    for u, v in zip(edges_df["from"].to_numpy(int), edges_df["to"].to_numpy(int)):
        adj[u].add(v)
        adj[v].add(u)
    adj2 = [set(adj[i]) for i in range(N)]
    for i in range(N):
        for nb in list(adj[i]):
            adj2[i].update(adj[nb])
        adj2[i].discard(i)

    tod = np.arange(T) % steps_per_day
    r2_records: List[float] = []
    n_spatial_imputed = 0
    n_fallback_imputed = 0

    for c in range(C):
        fault_c = fault_mask_3ch[:, :, c]
        if not fault_c.any():
            continue
        Xc = data[:, :, c].copy()
        Xc_nan = np.where(fault_c, np.nan, Xc)

        with np.errstate(all="ignore"):
            col_means = np.nanmean(Xc_nan, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        filled = np.where(np.isnan(Xc_nan), col_means[None, :], Xc_nan)
        centered = filled - filled.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0, keepdims=True) + 1e-8
        corr_mat = (centered.T @ centered) / (norms.T @ norms)
        np.fill_diagonal(corr_mat, -2.0)

        for i in range(N):
            bad_t = np.where(fault_c[:, i])[0]
            if len(bad_t) == 0:
                continue

            graph_cands = sorted(adj2[i], key=lambda j: -corr_mat[i, j])
            global_cands = list(np.argsort(-corr_mat[i])[: top_k * 2])
            cands: List[int] = []
            for j in graph_cands + global_cands:
                if j != i and j not in cands:
                    cands.append(int(j))
                if len(cands) >= top_k * 2:
                    break

            models: List[Tuple[int, float, float, float]] = []
            good_i = ~fault_c[:, i]
            for j in cands:
                joint = good_i & (~fault_c[:, j])
                if joint.sum() < 48:
                    continue
                xj = Xc[joint, j]
                xi = Xc[joint, i]
                vx = float(np.var(xj))
                vi = float(np.var(xi))
                if vx < 1e-8 or vi < 1e-8:
                    continue
                cov = float(np.mean((xj - xj.mean()) * (xi - xi.mean())))
                a = cov / vx
                b = float(xi.mean() - a * xj.mean())
                r2 = float((cov * cov) / (vx * vi))
                if r2 > 0.05:
                    models.append((j, a, b, r2))
                    r2_records.append(r2)
                if len(models) >= top_k:
                    break

            # Vectorized positive historical time-of-day median fallback (when neighbors fail or OLS < 0)
            D = T // steps_per_day
            pos_vals = Xc_nan[:, i]
            pos_only = pos_vals[(~np.isnan(pos_vals)) & (pos_vals > 0.0)]
            min_pos = float(np.percentile(pos_only, 1)) if len(pos_only) > 0 else (1.0 if c == 0 else (0.001 if c == 1 else 3.0))
            min_pos = max(min_pos, 1e-4 if c == 1 else 1.0)
            global_med = float(np.median(pos_only)) if len(pos_only) > 0 else min_pos
            with np.errstate(all="ignore"):
                prof = np.nanmedian(Xc_nan[: D * steps_per_day, i].reshape(D, steps_per_day), axis=0).astype(np.float32)
            prof = np.where(np.isnan(prof) | (prof <= 0.0), global_med, prof).astype(np.float32)

            if models:
                num = np.zeros(len(bad_t), dtype=np.float64)
                den = np.zeros(len(bad_t), dtype=np.float64)
                for j, a, b, r2 in models:
                    avail = ~fault_c[bad_t, j]
                    if avail.any():
                        pred = a * Xc[bad_t[avail], j] + b
                        # Prevent negative-intercept OLS clipping to artificial 0.0 on faulty slots
                        pred = np.maximum(pred, min_pos)
                        num[avail] += r2 * pred
                        den[avail] += r2
                has_est = den > 1e-8
                n_spatial_imputed += int(has_est.sum())
                n_fallback_imputed += int((~has_est).sum())
                est = np.where(has_est, num / np.maximum(den, 1e-8), prof[tod[bad_t]])
                est = np.maximum(est, min_pos)
                out[bad_t, i, c] = est.astype(np.float32)
            else:
                n_fallback_imputed += len(bad_t)
                out[bad_t, i, c] = np.maximum(prof[tod[bad_t]], min_pos)

    out = _clip_physical(out)
    meta = {
        "mean_pair_r2": float(np.mean(r2_records)) if r2_records else 0.0,
        "median_pair_r2": float(np.median(r2_records)) if r2_records else 0.0,
        "n_spatial_imputed": n_spatial_imputed,
        "n_fallback_imputed": n_fallback_imputed,
    }
    return out, meta


def _clip_physical(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, None)
    arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)
    arr[:, :, 2] = np.clip(arr[:, :, 2], 0.0, None)
    return arr


def evaluate_imputation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: np.ndarray,
    channel: int = 0,
) -> Dict[str, float]:
    """Compute MAE, RMSE, Pearson r, and 5-min Step-diff Std preservation on masked evaluation spans."""
    m = eval_mask[:, :, channel]
    yt = y_true[:, :, channel][m]
    yp = y_pred[:, :, channel][m]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 and np.std(yp) > 1e-8 else 0.0

    t_diffs, p_diffs = [], []
    _, N = m.shape
    for i in range(N):
        mi = m[:, i]
        if not mi.any():
            continue
        pad = np.concatenate([[False], mi, [False]])
        idx = np.where(pad[1:] != pad[:-1])[0]
        for a, b in zip(idx[0::2], idx[1::2]):
            if b - a >= 2:
                t_diffs.append(np.diff(y_true[a:b, i, channel]))
                p_diffs.append(np.diff(y_pred[a:b, i, channel]))
    gt_step_std = float(np.std(np.concatenate(t_diffs))) if t_diffs else 0.0
    pred_step_std = float(np.std(np.concatenate(p_diffs))) if p_diffs else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "r": r,
        "StepDiffStd_GT": gt_step_std,
        "StepDiffStd_Pred": pred_step_std,
        "Vol_Ratio": pred_step_std / max(gt_step_std, 1e-8),
    }
