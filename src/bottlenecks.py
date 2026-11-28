"""RQ1 Data-Driven Fundamental Diagram Threshold Calibration, 3-Phase Traffic Classification
(Chen 2004 + Choe 2002/HCM + Cassidy & Bertini 1999), and Detrended Pre-Congestion Signals
(Kerner 2004 + Critical Slowing Down — Chattopadhyay & Gupta 2026)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


def load_cleaned_pems(
    dataset_name: str,
    clean_dir: str | Path = "datasets/cleaned",
    raw_dir: str | Path = "datasets/data",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load Chen-2003-cleaned PeMS tensor (T, N, 3) [Flow, Occ, Speed] and deduplicated directed edges."""
    clean_path = Path(clean_dir) / f"{dataset_name}_clean.npz"
    csv_path = Path(raw_dir) / dataset_name / f"{dataset_name}.csv"
    npz_obj = np.load(clean_path)
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
class CongestionThresholds:
    """Data-driven traffic regime thresholds calibrated directly from a map's Fundamental Diagram."""
    v_ff: float       # Free-flow speed: median(Speed | Occ < 0.05)
    v_cong: float     # Congestion breakdown speed: 0.60 * v_ff (40% drop below free-flow, HCM/PeMS)
    v_sync: float     # Synchronized-flow onset speed: 0.80 * v_ff (20% drop below free-flow)
    o_crit: float     # Critical occupancy at peak mean Flow Q_max on Fundamental Diagram
    peak_flow: float  # Peak mean Flow Q_max (veh / 5 min)
    o_sat: float      # Saturation occupancy: earliest rising-branch bin reaching 90% of Q_max


def calibrate_congestion_thresholds(
    flow: np.ndarray,
    occ: np.ndarray,
    speed: np.ndarray,
    free_occ_max: float = 0.05,
    cong_speed_ratio: float = 0.60,
    sync_speed_ratio: float = 0.80,
    cap_ratio: float = 0.90,
    bin_width: float = 0.005,
    min_bin_count: int = 300,
) -> CongestionThresholds:
    """
    Compute data-driven congestion & synchronized-phase thresholds directly from the dataset's
    Fundamental Diagram (no hardcoded dataset-specific constants!):
      1. v_ff   = median(Speed | Occ < 0.05)
      2. v_cong = 0.60 * v_ff   (Congestion breakdown — mirrors ~40 mph on PeMS)
      3. v_sync = 0.80 * v_ff   (Synchronized transition onset)
      4. o_crit = argmax_o Q_smooth(o) on [0, 0.45]
      5. o_sat  = min{ o <= o_crit : Q_smooth(o) >= 0.90 * Q_max } (Pre-saturation floor, filters empty-road noise)
    """
    fl_r, oc_r, sp_r = flow.ravel(), occ.ravel(), speed.ravel()
    v_ff = float(np.median(sp_r[oc_r < free_occ_max]))
    v_cong = float(cong_speed_ratio * v_ff)
    v_sync = float(sync_speed_ratio * v_ff)

    grid = np.arange(0.0, 0.45, bin_width)
    c_occ, m_fl = [], []
    for o0 in grid:
        m = (oc_r >= o0) & (oc_r < o0 + bin_width)
        if int(m.sum()) >= min_bin_count:
            c_occ.append(float(o0 + 0.5 * bin_width))
            m_fl.append(float(fl_r[m].mean()))
    c_occ_arr = np.array(c_occ, dtype=np.float64)
    m_fl_arr = np.array(m_fl, dtype=np.float64)

    # 3-bin moving average to smooth bin sampling noise while preserving the capacity peak
    q_smooth = np.convolve(m_fl_arr, np.ones(3) / 3.0, mode="same")
    q_smooth[0] = m_fl_arr[0]
    q_smooth[-1] = m_fl_arr[-1]

    idx_peak = int(np.argmax(q_smooth[2:-2]) + 2)
    o_crit = float(c_occ_arr[idx_peak])
    peak_flow = float(q_smooth[idx_peak])

    rising_sat = np.where((np.arange(len(c_occ_arr)) <= idx_peak) & (q_smooth >= cap_ratio * peak_flow))[0]
    o_sat = float(c_occ_arr[rising_sat[0]]) if len(rising_sat) > 0 else o_crit

    return CongestionThresholds(
        v_ff=v_ff,
        v_cong=v_cong,
        v_sync=v_sync,
        o_crit=o_crit,
        peak_flow=peak_flow,
        o_sat=o_sat,
    )


def classify_traffic_phases(
    speed: np.ndarray,
    occ: np.ndarray,
    th: CongestionThresholds,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify every (t, node) observation into Kerner's 3 physical traffic phases using data-driven thresholds:
      - Phase 2 (Congestion / Breakdown): (Speed < v_cong) & (Occ >= o_sat)
      - Phase 1 (Synchronized Transition): not Phase 2 & ((Speed < v_sync) | (Occ >= o_sat))
      - Phase 0 (Free Flow): otherwise
    Returns: (phase_int8, is_congested_bool, is_synchronized_bool).
    """
    is_cong = (speed < th.v_cong) & (occ >= th.o_sat)
    is_sync = (~is_cong) & ((speed < th.v_sync) | (occ >= th.o_sat))
    phase = np.zeros(speed.shape, dtype=np.int8)
    phase[is_sync] = 1
    phase[is_cong] = 2
    return phase, is_cong, is_sync


def summarize_station_congestion(
    data: np.ndarray,
    edges_df: pd.DataFrame,
    is_congested: np.ndarray,
    is_synchronized: np.ndarray,
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Build clean per-station congestion summary table and directed graph for visualization & GNN input."""
    _, N, _ = data.shape
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i)
    for row in edges_df.itertuples(index=False):
        G.add_edge(int(row[0]), int(row[1]), weight=float(row[2]))

    cong_freq = is_congested.mean(axis=0)
    sync_freq = is_synchronized.mean(axis=0)
    mean_flow = data[:, :, 0].mean(axis=0)
    mean_occ = data[:, :, 1].mean(axis=0)
    mean_spd = data[:, :, 2].mean(axis=0)
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)

    rows = []
    for i in range(N):
        in_d = int(G.in_degree(i))
        out_d = int(G.out_degree(i))
        rows.append(
            {
                "node_id": i,
                "in_degree": in_d,
                "out_degree": out_d,
                "degree": in_d + out_d,
                "betweenness": float(bet[i]),
                "mean_flow": float(mean_flow[i]),
                "mean_occ": float(mean_occ[i]),
                "mean_speed": float(mean_spd[i]),
                "congestion_freq": float(cong_freq[i]),
                "sync_freq": float(sync_freq[i]),
            }
        )
        G.nodes[i]["congestion_freq"] = float(cong_freq[i])
        G.nodes[i]["sync_freq"] = float(sync_freq[i])
        G.nodes[i]["mean_speed"] = float(mean_spd[i])
        G.nodes[i]["betweenness"] = float(bet[i])

    return pd.DataFrame(rows), G


# ---------------------------------------------------------------------------
# Detrended Pre-Congestion Volatility & Critical Slowing Down Features
# (Kerner 2004 + Critical Slowing Down — Chattopadhyay & Gupta 2026)
# ---------------------------------------------------------------------------

CSD_FEATURE_NAMES = [
    "occ_rstd",           # Detrended rolling Std of Occupancy over H (Kerner/CSD volatility)
    "spd_rstd",           # Detrended rolling Std of Speed over H (stop-and-go oscillation)
    "flow_rstd",          # Detrended rolling Std of Flow over H
    "occ_spd_corr",       # Multivariate CSD: rolling -corr(Occ, Speed) over H
    "flow_occ_decouple",  # Multivariate CSD: rolling -corr(Flow, Occ) over H (Fundamental Diagram breakdown)
    "occ_slope",          # Signed linear slope of Occupancy per 5-min step (+ buildup, - clearing)
    "spd_slope",          # Signed linear slope of Speed per 5-min step (- breakdown, + recovery)
]


def _causal_detrend_stats(x: np.ndarray, window: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal (backward-looking over [t - window + 1, t]) linear detrending:
      1. Detrended residual Std (pure fluctuation without ramp bias),
      2. Signed linear regression slope per 5-min step (separates breakdown from recovery).
    """
    T, N = x.shape
    rstd = np.zeros((T, N), dtype=np.float32)
    slope = np.zeros((T, N), dtype=np.float32)
    if T < window:
        return rstd, slope

    t_idx = np.arange(window, dtype=np.float64)
    t_c = t_idx - t_idx.mean()
    t_denom = float((t_c ** 2).sum())

    w = sliding_window_view(x.astype(np.float64), window_shape=window, axis=0)
    mean = w.mean(axis=-1, keepdims=True)
    sl = ((w - mean) * t_c).sum(axis=-1) / t_denom
    resid = w - (mean + sl[..., None] * t_c)

    rstd[window - 1 :] = resid.std(axis=-1).astype(np.float32)
    slope[window - 1 :] = sl.astype(np.float32)
    return rstd, slope


def _causal_neg_cross_corr(x: np.ndarray, y: np.ndarray, window: int = 12) -> np.ndarray:
    """Strictly causal rolling -corr(x, y) over [t - window + 1, t] (Multivariate CSD coupling)."""
    T, N = x.shape
    out = np.zeros((T, N), dtype=np.float32)
    if T < window:
        return out
    wx = sliding_window_view(x.astype(np.float64), window_shape=window, axis=0)
    wy = sliding_window_view(y.astype(np.float64), window_shape=window, axis=0)
    cx = wx - wx.mean(axis=-1, keepdims=True)
    cy = wy - wy.mean(axis=-1, keepdims=True)
    num = -(cx * cy).sum(axis=-1)
    den = np.sqrt((cx * cx).sum(axis=-1) * (cy * cy).sum(axis=-1))
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    out[window - 1 :] = np.clip(corr, -1.0, 1.0).astype(np.float32)
    return out


def extract_precongestion_features(data: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Compute strictly causal 7-channel Detrended Volatility, Multivariate CSD Coupling, and Signed Slope tensor
    of shape (T, N, 7) matching CSD_FEATURE_NAMES over rolling window H (default 12 = 60 min).
    Can be directly concatenated with [Flow, Occ, Speed] as input features for RQ2 models.
    """
    flow_rstd, _ = _causal_detrend_stats(data[:, :, 0], window=window)
    occ_rstd, occ_slope = _causal_detrend_stats(data[:, :, 1], window=window)
    spd_rstd, spd_slope = _causal_detrend_stats(data[:, :, 2], window=window)
    occ_spd_corr = _causal_neg_cross_corr(data[:, :, 1], data[:, :, 2], window=window)
    flow_occ_decouple = _causal_neg_cross_corr(data[:, :, 0], data[:, :, 1], window=window)
    return np.stack(
        [occ_rstd, spd_rstd, flow_rstd, occ_spd_corr, flow_occ_decouple, occ_slope, spd_slope],
        axis=-1,
    ).astype(np.float32)


def merge_phase_spans(x_vals: np.ndarray, mask: np.ndarray, half_step: float) -> List[Tuple[float, float]]:
    """Merge contiguous boolean steps into solid [start, end] spans for clean chart shading."""
    spans: List[Tuple[float, float]] = []
    in_seg, start = False, 0.0
    for k, flag in enumerate(mask):
        if flag and not in_seg:
            start, in_seg = float(x_vals[k] - half_step), True
        elif not flag and in_seg:
            spans.append((start, float(x_vals[k - 1] + half_step)))
            in_seg = False
    if in_seg:
        spans.append((start, float(x_vals[-1] + half_step)))
    return spans


def evaluate_precongestion_signals(
    data: np.ndarray,
    is_congested: np.ndarray,
    is_synchronized: np.ndarray,
    csd: np.ndarray,
    window: int = 12,
    post_steps: int = 18,
    lead_univar: int = 3,
    seed: int = 42,
) -> dict:
    """
    Run clean full-network RQ1 evaluation of the 7 pre-congestion signals on fresh breakdowns
    (zero congestion in prior `window` steps):
      - Full-cycle mean trajectory over [-window .. +post_steps] (default -60m .. +90m, covering onset + recovery)
      - Univariate ROC-AUC & Top-10% Risk Ratio (all clean-60m vs clean-60m & Speed >= 55 mph)
      - Added-value Likelihood-Ratio Test (60/20/20 split, Raw 3 vs Raw + 7 CSD) for dt in {5, 15, 30}m.
    """
    from scipy.stats import chi2, mannwhitneyu
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    T, N, _ = data.shape
    flow, occ, spd = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    cg_past = np.vstack([np.ones((window - 1, N), dtype=bool), sliding_window_view(is_congested, window, axis=0).any(axis=-1)])

    # 1. Fresh onsets (no congestion in prior 60 min) with full post-onset recovery window
    all_on, fresh_on = [], []
    for u in range(N):
        ev = np.where(is_congested[window : T - post_steps, u] & (~is_congested[window - 1 : T - post_steps - 1, u]))[0] + window
        for t0 in ev:
            all_on.append((int(t0), u))
            if (~is_congested[t0 - window : t0, u]).all():
                fresh_on.append((int(t0), u))

    rel_steps = np.arange(-window, post_steps + 1)
    curves = {m: [] for m in ["flow", "occ", "spd", *CSD_FEATURE_NAMES]}
    for off in rel_steps:
        curves["flow"].append(float(np.mean([flow[t0 + off, u] for t0, u in fresh_on])))
        curves["occ"].append(float(np.mean([occ[t0 + off, u] for t0, u in fresh_on])))
        curves["spd"].append(float(np.mean([spd[t0 + off, u] for t0, u in fresh_on])))
        for fi, fn in enumerate(CSD_FEATURE_NAMES):
            curves[fn].append(float(np.mean([csd[t0 + off, u, fi] for t0, u in fresh_on])))

    # 2. Univariate AUC & Risk Ratio (dt = lead_univar)
    m_clean = ~cg_past[window : T - lead_univar]
    m_fast = m_clean & (spd[window : T - lead_univar] >= 55.0)
    y_c = is_congested[window + lead_univar :][m_clean].astype(int)
    y_f = is_congested[window + lead_univar :][m_fast].astype(int)

    csd_rows = []
    for fi, fn in enumerate(CSD_FEATURE_NAMES):
        vc = csd[window : T - lead_univar, :, fi][m_clean]
        vf = csd[window : T - lead_univar, :, fi][m_fast]
        sc = -vc if fn == "spd_slope" else vc
        sf = -vf if fn == "spd_slope" else vf
        auc_c, auc_f = float(roc_auc_score(y_c, sc)), float(roc_auc_score(y_f, sf))
        q90 = float(np.percentile(sc, 90))
        rr_c = float(y_c[sc >= q90].mean() / max(y_c[sc < q90].mean(), 1e-9))
        _, pv = mannwhitneyu(vc[y_c == 1], vc[y_c == 0], alternative="two-sided")
        mp, mn = float(np.median(vc[y_c == 1])), float(np.median(vc[y_c == 0]))
        csd_rows.append({
            "Chỉ báo (H = 60p)": fn,
            "Trung vị Sắp tắc": round(mp, 4),
            "Trung vị Thông": round(mn, 4),
            "Hệ số tăng": f"{mp / mn:.2f}x" if (mn > 1e-5 and mp > 0) else "Đổi dấu",
            "AUC (Thông sạch 60p)": round(auc_c, 4),
            "RR Top 10%": f"{rr_c:.2f}x",
            "AUC (Speed ≥ 55 mph)": round(auc_f, 4),
            "p (Mann–Whitney)": "< 1e-300" if pv == 0.0 else f"{pv:.2e}",
        })

    # 3. Added-value LRT (60/20/20 chronological split)
    t_tr, t_va = int(T * 0.60), int(T * 0.80)
    lrt_rows = []
    for lead in [1, 3, 6]:
        def _sl(a, b):
            fn_m = ~cg_past[a:b]
            yy = is_congested[a + lead : b + lead][fn_m].astype(int)
            xr = data[a:b][fn_m]
            xc = csd[a:b][fn_m]
            return xr, np.hstack([xr, xc]), yy

        Xr_tr, Xb_tr, y_tr = _sl(window, t_tr - lead)
        Xr_va, Xb_va, y_va = _sl(t_tr, t_va - lead)
        sc_r, sc_b = StandardScaler().fit(Xr_tr), StandardScaler().fit(Xb_tr)
        clf_r = LogisticRegression(max_iter=500, random_state=seed).fit(sc_r.transform(Xr_tr), y_tr)
        clf_b = LogisticRegression(max_iter=500, random_state=seed).fit(sc_b.transform(Xb_tr), y_tr)
        pr_r = clf_r.predict_proba(sc_r.transform(Xr_va))[:, 1]
        pr_b = clf_b.predict_proba(sc_b.transform(Xb_va))[:, 1]
        ap_r, ap_b = float(average_precision_score(y_va, pr_r)), float(average_precision_score(y_va, pr_b))
        ll_r, ll_b = float(log_loss(y_va, pr_r)), float(log_loss(y_va, pr_b))
        c2 = float(2.0 * len(y_va) * max(0.0, ll_r - ll_b))
        pv = float(chi2.sf(c2, df=len(CSD_FEATURE_NAMES)))
        lrt_rows.append({
            "Tầm cảnh báo (Δt)": f"{lead*5} phút",
            "Raw 3 biến (Val PR-AUC)": round(ap_r, 4),
            "Raw + 7 CSD (Val PR-AUC)": round(ap_b, 4),
            "Tăng PR-AUC (%)": f"{(ap_b / ap_r - 1.0) * 100:+.2f}%",
            "Log-Loss (Raw → +CSD)": f"{ll_r:.5f} → {ll_b:.5f}",
            "LRT χ² (df=7)": round(c2, 1),
            "p (LRT)": f"{pv:.2e}",
        })

    return {
        "n_all_onsets": len(all_on),
        "n_fresh_onsets": len(fresh_on),
        "rel_steps": rel_steps,
        "curves": curves,
        "csd_df": pd.DataFrame(csd_rows),
        "lrt_df": pd.DataFrame(lrt_rows),
    }


def evaluate_dynamic_graph_signals(
    data: np.ndarray,
    edges_df: pd.DataFrame,
    is_congested: np.ndarray,
    csd: np.ndarray,
    window: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Quantify dynamic directed-graph contribution (upstream In & downstream Out neighbor states
    [Flow, Occ, Speed]) on top of local [3 raw + 7 pre-congestion] features via 60/20/20 LRT.
    Demonstrates that while static graph topology (Betweenness) fails, dynamic neighbor states
    flowing along directed edges significantly improve congestion forecasting (DCRNN principle).
    """
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, log_loss
    from sklearn.preprocessing import StandardScaler

    T, N, _ = data.shape
    a_in = np.zeros((N, N), dtype=np.float32)
    a_out = np.zeros((N, N), dtype=np.float32)
    for row in edges_df.itertuples(index=False):
        u, v = int(row[0]), int(row[1])
        a_in[v, u] = 1.0
        a_out[u, v] = 1.0

    w_in = np.divide(a_in, a_in.sum(1, keepdims=True), out=np.zeros_like(a_in), where=a_in.sum(1, keepdims=True) > 0)
    w_out = np.divide(a_out, a_out.sum(1, keepdims=True), out=np.zeros_like(a_out), where=a_out.sum(1, keepdims=True) > 0)
    nbr_in = np.einsum("ij,tjf->tif", w_in, data)
    nbr_out = np.einsum("ij,tjf->tif", w_out, data)
    nbr_both = np.concatenate([nbr_in, nbr_out], axis=-1)

    cg_past = np.vstack([np.ones((window - 1, N), dtype=bool), sliding_window_view(is_congested, window, axis=0).any(axis=-1)])
    t_tr, t_va = int(T * 0.60), int(T * 0.80)

    def _fit_eval(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray) -> Tuple[float, float]:
        sc = StandardScaler().fit(x_tr)
        clf = LogisticRegression(max_iter=500, random_state=seed).fit(sc.transform(x_tr), y_tr)
        pr = clf.predict_proba(sc.transform(x_va))[:, 1]
        return float(average_precision_score(y_va, pr)), float(log_loss(y_va, pr))

    rows = []
    for lead in [1, 3, 6]:
        m_tr = ~cg_past[window : t_tr - lead]
        m_va = ~cg_past[t_tr : t_va - lead]
        y_tr = is_congested[window + lead : t_tr][m_tr].astype(int)
        y_va = is_congested[t_tr + lead : t_va][m_va].astype(int)

        xl_tr = np.hstack([data[window : t_tr - lead][m_tr], csd[window : t_tr - lead][m_tr]])
        xl_va = np.hstack([data[t_tr : t_va - lead][m_va], csd[t_tr : t_va - lead][m_va]])
        xi_tr = np.hstack([xl_tr, nbr_in[window : t_tr - lead][m_tr]])
        xi_va = np.hstack([xl_va, nbr_in[t_tr : t_va - lead][m_va]])
        xo_tr = np.hstack([xl_tr, nbr_out[window : t_tr - lead][m_tr]])
        xo_va = np.hstack([xl_va, nbr_out[t_tr : t_va - lead][m_va]])
        xb_tr = np.hstack([xl_tr, nbr_both[window : t_tr - lead][m_tr]])
        xb_va = np.hstack([xl_va, nbr_both[t_tr : t_va - lead][m_va]])

        ap_l, ll_l = _fit_eval(xl_tr, y_tr, xl_va, y_va)
        ap_i, _ = _fit_eval(xi_tr, y_tr, xi_va, y_va)
        ap_o, _ = _fit_eval(xo_tr, y_tr, xo_va, y_va)
        ap_b, ll_b = _fit_eval(xb_tr, y_tr, xb_va, y_va)

        c2 = float(2.0 * len(y_va) * max(0.0, ll_l - ll_b))
        pv = float(chi2.sf(c2, df=6))
        rows.append({
            "Tầm cảnh báo (Δt)": f"{lead * 5} phút",
            "Cục bộ (3 gốc + 7 chỉ báo)": round(ap_l, 4),
            "+ Lân cận In (Đầu vào)": round(ap_i, 4),
            "+ Lân cận Out (Đầu ra)": round(ap_o, 4),
            "+ Cả 2 chiều (In + Out)": round(ap_b, 4),
            "Tăng PR-AUC (%)": f"{(ap_b / ap_l - 1.0) * 100:+.2f}%",
            "LRT χ² (df=6)": round(c2, 1),
            "p (LRT)": f"{pv:.2e}",
        })
    return pd.DataFrame(rows)
