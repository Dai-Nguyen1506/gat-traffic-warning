"""Publication-grade 300 DPI plotting utilities for NB04 (RQ2 on PEMS08 with ST-GAT LSTM -> GATv2)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bottlenecks import CongestionThresholds, merge_phase_spans

PAL = {
    "navy": "#1B365D",
    "blue": "#2B6CB0",
    "coral": "#D64545",
    "amber": "#E69F00",
    "teal": "#1E8E7E",
    "purple": "#7C6BA8",
    "slate": "#475569",
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
}

GROUP_COLORS = {
    "Raw (3 biến gốc)": PAL["slate"],
    "Nhóm 4 — STID (Chu kỳ thời gian)": PAL["purple"],
    "Nhóm 1 — Dao động khử dốc": PAL["blue"],
    "Nhóm 2 — Liên kết đa biến": PAL["teal"],
    "Nhóm 3 — Độ dốc có dấu": PAL["amber"],
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "savefig.dpi": 300,
            "figure.dpi": 115,
            "font.size": 10,
        }
    )


def _save_and_show(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu biểu đồ: {save_path.name}")


def plot_feature_ablation_mae(
    ablation_df: pd.DataFrame,
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Path,
) -> None:
    """Hình 1 (2×2): Đối chứng MAE của 6 cấu hình đặc trưng trên ST-GAT (LSTM -> GATv2) & Đường cong học."""
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.2))
    cfgs = ablation_df["Cấu hình"].tolist()
    short_lbls = [
        c.replace("Base + Nhóm ", "+G").replace(" (", "\n(") for c in cfgs
    ]
    x = np.arange(len(cfgs))
    bar_cols = ["#CBD5E1", PAL["purple"], PAL["blue"], PAL["teal"], PAL["amber"], PAL["navy"]]

    # (A) MAE Speed across all 6 configs (Avg 5-60p)
    ax = axes[0, 0]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals = ablation_df["MAE Speed (mph)"].to_numpy()
    bars = ax.bar(x, vals, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_v = vals[1] if len(vals) > 1 else vals[0]
    for i, (b, v) in enumerate(zip(bars, vals)):
        delta = (v - base_v) / base_v * 100.0
        tag = f"{v:.3f}\n({delta:+.1f}%)" if i >= 2 else f"{v:.3f}"
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (vals.max() - vals.min()) * 0.04 + 0.015,
            tag,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if i == len(vals) - 1 else "normal",
            color=PAL["navy"] if i == len(vals) - 1 else PAL["slate"],
        )
    ax.axhline(base_v, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ymin = max(0.0, vals.min() * 0.88)
    ymax = vals.max() * 1.12
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("Test MAE Speed (mph) — TB 5–60p")
    ax.set_title("(A) Sai số Vận tốc toàn mạng (MAE Speed, thấp hơn là tốt hơn)", fontweight="bold")

    # (B) Normalized Std-MAE (3 variables combined) across all 6 configs
    ax = axes[0, 1]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals_s = ablation_df["Std-MAE (3 biến)"].to_numpy()
    bars_s = ax.bar(x, vals_s, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_s = vals_s[1] if len(vals_s) > 1 else vals_s[0]
    for i, (b, v) in enumerate(zip(bars_s, vals_s)):
        delta = (v - base_s) / base_s * 100.0
        tag = f"{v:.4f}\n({delta:+.1f}%)" if i >= 2 else f"{v:.4f}"
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (vals_s.max() - vals_s.min()) * 0.04 + 0.001,
            tag,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if i == len(vals_s) - 1 else "normal",
            color=PAL["navy"] if i == len(vals_s) - 1 else PAL["slate"],
        )
    ax.axhline(base_s, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ax.set_ylim(max(0.0, vals_s.min() * 0.88), vals_s.max() * 1.12)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("Test Std-MAE (trung bình chuẩn hóa 3 biến)")
    ax.set_title("(B) Sai số tổng hợp 3 biến (Flow + Occ + Speed, % so với Base)", fontweight="bold")

    # (C) Bottleneck stations Speed MAE (Train-calibrated >= 1% bottlenecks)
    ax = axes[1, 0]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals_c = ablation_df["MAE Speed Điểm nghẽn (mph)"].to_numpy()
    bars_c = ax.bar(x, vals_c, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_c = vals_c[1] if len(vals_c) > 1 else vals_c[0]
    for i, (b, v) in enumerate(zip(bars_c, vals_c)):
        delta = (v - base_c) / base_c * 100.0
        tag = f"{v:.3f}\n({delta:+.1f}%)" if i >= 2 else f"{v:.3f}"
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (vals_c.max() - vals_c.min()) * 0.04 + 0.015,
            tag,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if i == len(vals_c) - 1 else "normal",
            color=PAL["coral"] if i == len(vals_c) - 1 else PAL["slate"],
        )
    ax.axhline(base_c, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ax.set_ylim(max(0.0, vals_c.min() * 0.88), vals_c.max() * 1.12)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("Test MAE Speed tại trạm Điểm nghẽn (mph)")
    ax.set_title("(C) Sai số Vận tốc trên nhóm trạm Điểm nghẽn (xác định thuần Train)", fontweight="bold")

    # (D) Validation L1 learning curves for Raw(3), Base(7), Full(14)
    ax = axes[1, 1]
    ax.grid(True, alpha=0.65)
    curve_specs = [
        ("Raw (3)", PAL["muted"], "--", "o"),
        ("Base (7)", PAL["purple"], "-.", "s"),
        ("Full (14)", PAL["navy"], "-", "^"),
    ]
    for cname, col, ls, mk in curve_specs:
        if cname in histories and len(histories[cname].get("val_loss", [])) > 0:
            vl = histories[cname]["val_loss"]
            ep = np.arange(1, len(vl) + 1)
            ax.plot(ep, vl, color=col, ls=ls, lw=2.1, marker=mk, ms=4.0, label=f"{cname} (best={min(vl):.4f})")
    ax.set_xlabel("Epoch huấn luyện")
    ax.set_ylabel("Validation L1 Loss (Std-MAE)")
    ax.set_title("(D) Quá trình hội tụ Validation L1 Loss (SEED = 42)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], fontsize=8.4)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    _save_and_show(fig, save_path)


def plot_horizon_and_congestion(
    horizon_curves: Dict[str, Dict[str, np.ndarray]],
    cong_check_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Hình 2 (1×3): Sai số MAE theo đủ 12 tầm dự báo (5p → 60p) & Đánh giá phụ đánh dấu tắc nghẽn."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    steps_min = np.arange(1, 13) * 5

    specs = [
        ("Raw (3)", PAL["muted"], "--", "o"),
        ("Base (7)", PAL["purple"], "-.", "s"),
        ("Full (14)", PAL["navy"], "-", "^"),
    ]

    # (A) Network-wide MAE Speed across 5..60 min
    ax = axes[0]
    ax.grid(True, alpha=0.65)
    for cname, col, ls, mk in specs:
        if cname in horizon_curves:
            ax.plot(steps_min, horizon_curves[cname]["mae_speed"], color=col, ls=ls, lw=2.2, marker=mk, ms=4.2, label=cname)
    ax.set_xticks([5, 15, 30, 45, 60])
    ax.set_xlabel("Tầm dự báo tương lai Δt (phút)")
    ax.set_ylabel("Test MAE Speed (mph)")
    ax.set_title("(A) MAE Vận tốc toàn mạng (5p → 60p)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], fontsize=8.4)

    # (B) Network-wide MAE Flow across 5..60 min
    ax = axes[1]
    ax.grid(True, alpha=0.65)
    for cname, col, ls, mk in specs:
        if cname in horizon_curves:
            ax.plot(steps_min, horizon_curves[cname]["mae_flow"], color=col, ls=ls, lw=2.2, marker=mk, ms=4.2, label=cname)
    ax.set_xticks([5, 15, 30, 45, 60])
    ax.set_xlabel("Tầm dự báo tương lai Δt (phút)")
    ax.set_ylabel("Test MAE Flow (xe / 5 phút)")
    ax.set_title("(B) MAE Lưu lượng toàn mạng (5p → 60p)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], fontsize=8.4)

    # (C) Secondary congestion marking check (F1 at 15p / 30p / 60p)
    ax = axes[2]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    h_lbls = cong_check_df["Tầm dự báo"].tolist()
    xh = np.arange(len(h_lbls))
    w = 0.25
    for idx_c, (cname, col) in enumerate([("Raw (3)", "#CBD5E1"), ("Base (7)", PAL["purple"]), ("Full (14)", PAL["navy"])]):
        col_key = f"F1 {cname}"
        if col_key in cong_check_df.columns:
            vs = cong_check_df[col_key].to_numpy()
            bars = ax.bar(xh + (idx_c - 1) * w, vs, width=w * 0.94, color=col, edgecolor="#64748B", lw=0.5, label=cname)
            for b, v in zip(bars, vs):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold" if idx_c == 2 else "normal")
    ax.set_xticks(xh, h_lbls)
    ax.set_ylim(0, min(1.05, cong_check_df[[c for c in cong_check_df.columns if c.startswith("F1 ")]].to_numpy().max() * 1.20))
    ax.set_ylabel("F1-Score đánh dấu tắc (ngưỡng Train)")
    ax.set_title("(C) Đánh giá phụ: Đánh dấu tắc nghẽn", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=8.2)

    fig.tight_layout(w_pad=2.0)
    _save_and_show(fig, save_path)


def plot_permutation_importance(
    group_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Hình 3 (1×2): Permutation Feature Importance (ΔStd-MAE) cho 4 nhóm đặc trưng bổ sung & 11 đặc trưng bổ sung."""
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), gridspec_kw={"width_ratios": [1.05, 1.25]})

    raw_grp_row = group_df[group_df["group"].str.startswith("Raw")]
    raw_grp_val = float(raw_grp_row["delta_std_mae"].iloc[0]) if len(raw_grp_row) else 0.0

    # (A) 4 Engineered Feature Groups (readable linear scale; Raw 3 anchor noted cleanly in box)
    ax = axes[0]
    ax.grid(axis="x", alpha=0.65)
    ax.set_axisbelow(True)
    gdf = group_df[~group_df["group"].str.startswith("Raw")].sort_values("delta_std_mae", ascending=True).reset_index(drop=True)
    y_g = np.arange(len(gdf))
    cols_g = [GROUP_COLORS.get(g, PAL["navy"]) for g in gdf["group"]]
    ax.barh(
        y_g,
        gdf["delta_std_mae"],
        xerr=gdf["delta_std_mae_sd"],
        height=0.52,
        color=cols_g,
        alpha=0.90,
        edgecolor="white",
        capsize=3,
    )
    xmax_g = float((gdf["delta_std_mae"] + gdf["delta_std_mae_sd"]).max())
    for i, row in gdf.iterrows():
        ax.text(
            row["delta_std_mae"] + row["delta_std_mae_sd"] + xmax_g * 0.025,
            i,
            f"+{row['delta_std_mae']:.4f}  (Spd +{row['delta_mae_speed']:.3f})",
            va="center",
            fontsize=8.3,
            fontweight="bold" if i >= len(gdf) - 2 else "normal",
            color=PAL["navy"] if i >= len(gdf) - 2 else PAL["slate"],
        )
    ax.text(
        0.97,
        0.10,
        f"Mốc neo Raw (3 biến gốc):\nΔStd-MAE = +{raw_grp_val:.4f}\n(tách riêng để không nén thang đo 4 nhóm bổ sung)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color=PAL["slate"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec=PAL["grid"]),
    )
    ax.set_yticks(y_g, [g.replace(" — ", "\n") for g in gdf["group"]], fontsize=8.8)
    ax.set_xlim(0, xmax_g * 1.56)
    ax.set_xlabel("Mức tăng Std-MAE khi xáo trộn nhóm (ΔStd-MAE ± SD, 5 lần lặp)")
    ax.set_title("(A) Đóng góp 4 nhóm đặc trưng bổ sung (Test PFI)", fontweight="bold")

    # (B) 11 Engineered Individual Features (4 STID + 7 Pre-Congestion Signals)
    ax = axes[1]
    ax.grid(axis="x", alpha=0.65)
    ax.set_axisbelow(True)
    fdf_raw = feature_df[feature_df["group"].str.startswith("Raw")].sort_values("delta_std_mae", ascending=False)
    fdf = feature_df[~feature_df["group"].str.startswith("Raw")].sort_values("delta_std_mae", ascending=True).reset_index(drop=True)
    y_f = np.arange(len(fdf))
    cols_f = [GROUP_COLORS.get(g, PAL["navy"]) for g in fdf["group"]]
    ax.barh(
        y_f,
        fdf["delta_std_mae"],
        xerr=fdf["delta_std_mae_sd"],
        height=0.64,
        color=cols_f,
        alpha=0.90,
        edgecolor="white",
        capsize=2.5,
    )
    xmax_f = float((fdf["delta_std_mae"] + fdf["delta_std_mae_sd"]).max())
    for i, row in fdf.iterrows():
        ax.text(
            max(0.0, row["delta_std_mae"] + row["delta_std_mae_sd"]) + xmax_f * 0.02,
            i,
            f"+{row['delta_std_mae']:.4f} (Spd +{row['delta_mae_speed']:.3f})",
            va="center",
            fontsize=8.0,
            fontweight="bold" if row["feature"] in ("tod_cos", "tod_sin", "occ_slope", "spd_slope", "occ_rstd") else "normal",
            color=PAL["navy"] if row["feature"] in ("tod_cos", "tod_sin", "occ_slope", "spd_slope") else PAL["slate"],
        )
    raw_lines = " | ".join([f"{r.feature}: +{r.delta_std_mae:.3f}" for r in fdf_raw.itertuples(index=False)])
    ax.text(
        0.97,
        0.05,
        f"3 biến đích gốc (tách thang):\n{raw_lines}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color=PAL["slate"],
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec=PAL["grid"]),
    )
    ax.set_yticks(y_f, fdf["feature"], fontsize=8.8)
    ax.set_xlim(0, xmax_f * 1.48)
    ax.set_xlabel("Mức tăng Std-MAE khi xáo trộn từng đặc trưng (ΔStd-MAE ± SD)")
    ax.set_title("(B) Xếp hạng 11 đặc trưng bổ sung trên ST-GAT Full (14)", fontweight="bold")

    fig.tight_layout(w_pad=2.2)
    _save_and_show(fig, save_path)


def plot_gatv2_attention_and_case(
    attn_res: Dict[str, object],
    y_true_test: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_full: np.ndarray,
    t_test: np.ndarray,
    train_thresholds: CongestionThresholds,
    horizon_step: int = 3,
    save_path: Path = Path("outputs/figures/nb04_gatv2_attention_and_case.png"),
) -> None:
    """Hình 4 (2 tầng): Chú ý cạnh GATv2 (Self / In / Out) & Case study 48 giờ tại điểm nghẽn trọng yếu."""
    fig = plt.figure(figsize=(15.4, 8.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.02], hspace=0.34, wspace=0.24)
    ax_agg = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_case = fig.add_subplot(gs[1, :])

    # (A) Aggregated Self / In / Out attention across all Train-calibrated bottlenecks
    sm = attn_res["summary_means"]
    cats = ["Self\n(Bản thân)", "In\n(Thượng nguồn)", "Out\n(Hạ nguồn dội ngược)"]
    norm_v = [sm["self_normal"], sm["in_normal"], sm["out_normal"]]
    cong_v = [sm["self_pre_cong"], sm["in_pre_cong"], sm["out_pre_cong"]]
    xc = np.arange(3)
    w = 0.34
    ax_agg.grid(axis="y", alpha=0.65)
    ax_agg.set_axisbelow(True)
    b1 = ax_agg.bar(xc - w / 2, norm_v, width=w, color="#CBD5E1", edgecolor="#94A3B8", lw=0.6, label="Bình thường (C = 0)")
    b2 = ax_agg.bar(xc + w / 2, cong_v, width=w, color=PAL["navy"], label="Tiền tắc / Sắp tắc (C = 1)")
    for b in list(b1) + list(b2):
        ax_agg.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax_agg.set_xticks(xc, cats)
    ax_agg.set_ylim(0, max(norm_v + cong_v) * 1.24)
    ax_agg.set_ylabel("Tổng trọng số chú ý cạnh TB (∑α_ij)")
    ax_agg.set_title(f"(A) Phân bổ chú ý GATv2 theo hướng cạnh ({len(attn_res['bn_summary_df'])} trạm điểm nghẽn)", fontweight="bold")
    ax_agg.legend(frameon=False, loc="upper right", fontsize=8.4)

    # (B) Per-edge attention for primary bottleneck node
    top_node = attn_res["top_node"]
    tdf = attn_res["top_edges_df"].copy()
    edge_lbls = [
        f"N{int(r.src)} → N{int(r.dst)}\n({r.direction.split()[0]})" for r in tdf.itertuples(index=False)
    ]
    ye = np.arange(len(tdf))
    ax_top.grid(axis="x", alpha=0.65)
    ax_top.set_axisbelow(True)
    h_bar = 0.36
    ax_top.barh(ye + h_bar / 2, tdf["alpha_normal"], height=h_bar, color="#CBD5E1", edgecolor="#94A3B8", lw=0.5, label="Bình thường")
    ax_top.barh(ye - h_bar / 2, tdf["alpha_pre_cong"], height=h_bar, color=PAL["coral"], alpha=0.88, label="Tiền tắc nghẽn")
    xmax_e = float(max(tdf["alpha_normal"].max(), tdf["alpha_pre_cong"].max()))
    for i, r in tdf.iterrows():
        ax_top.text(
            max(r["alpha_normal"], r["alpha_pre_cong"]) + xmax_e * 0.02,
            i,
            f"{r['alpha_shift']:+.4f}",
            va="center",
            fontsize=8.0,
            fontweight="bold" if r["alpha_shift"] > 0 else "normal",
            color=PAL["coral"] if r["alpha_shift"] > 0 else PAL["slate"],
        )
    ax_top.set_yticks(ye, edge_lbls, fontsize=8.0)
    ax_top.invert_yaxis()
    ax_top.set_xlim(0, xmax_e * 1.28)
    ax_top.set_xlabel("Trọng số chú ý α_ij (kèm độ dịch chuyển Δα)")
    ax_top.set_title(f"(B) Chi tiết trọng số chú ý các cạnh vào điểm nghẽn N{top_node}", fontweight="bold")
    ax_top.legend(frameon=False, loc="lower right", fontsize=8.2)

    # (C) 48-hour Test case study at top_node for horizon_step (default 15p)
    h_idx = horizon_step - 1
    sp_true = y_true_test[:, top_node, h_idx, 2]
    oc_true = y_true_test[:, top_node, h_idx, 1]
    sp_base = y_pred_base[:, top_node, h_idx, 2]
    sp_full = y_pred_full[:, top_node, h_idx, 2]
    cg_true = (sp_true < train_thresholds.v_cong) & (oc_true >= train_thresholds.o_sat)

    # Pick the 48h (576 steps) Test window with the richest congestion activity
    win = min(576, len(sp_true))
    best_start, best_cnt = 0, -1
    for s in range(0, len(sp_true) - win + 1, 24):
        cnt = int(cg_true[s : s + win].sum())
        if cnt > best_cnt:
            best_cnt, best_start = cnt, s
    sl = slice(best_start, best_start + win)
    hrs = np.arange(win) * 5.0 / 60.0

    for i_s, (a, b_sp) in enumerate(merge_phase_spans(hrs, cg_true[sl], 2.5 / 60.0)):
        ax_case.axvspan(a, b_sp, color=PAL["coral"], alpha=0.14, lw=0, label="Tắc thực (ngưỡng thuần Train)" if i_s == 0 else None)
    ax_case.plot(hrs, sp_true[sl], color="#0F172A", lw=1.9, label="Vận tốc thực (Ground Truth)")
    ax_case.plot(hrs, sp_base[sl], color=PAL["purple"], lw=1.5, ls="--", alpha=0.90, label=f"Dự báo Base (7) — MAE={np.abs(sp_base[sl]-sp_true[sl]).mean():.2f} mph")
    ax_case.plot(hrs, sp_full[sl], color=PAL["teal"], lw=1.7, alpha=0.95, label=f"Dự báo Full (14) — MAE={np.abs(sp_full[sl]-sp_true[sl]).mean():.2f} mph")
    ax_case.axhline(train_thresholds.v_cong, color=PAL["coral"], ls=":", lw=1.1, label=f"v_cong (Train) = {train_thresholds.v_cong:.2f} mph")
    ax_case.grid(axis="y", alpha=0.65)
    ax_case.set_xlim(0, hrs[-1])
    ax_case.set_ylim(12, 87)
    ax_case.set_xlabel(f"Cửa sổ 48 giờ trên tập Test (trạm N{top_node}, tầm dự báo Δt = {horizon_step * 5} phút)")
    ax_case.set_ylabel("Vận tốc (mph)")
    ax_case.set_title(f"(C) Minh họa dự báo vận tốc 48 giờ cao điểm trên tập Test tại N{top_node} (Base 7 vs. Full 14)", fontweight="bold", pad=10)
    ax_case.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.2)

    _save_and_show(fig, save_path)
