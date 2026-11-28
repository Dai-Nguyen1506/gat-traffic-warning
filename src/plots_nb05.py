"""Publication-grade 300 DPI plotting utilities for NB05 (Symmetric 6-Config Cross-Map Validation on PEMS04)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bottlenecks import CongestionThresholds

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


def _bin_fd(data_tr: np.ndarray, bin_width: float = 0.01, min_cnt: int = 200):
    fl = data_tr[:, :, 0].ravel()
    oc = data_tr[:, :, 1].ravel()
    sp = data_tr[:, :, 2].ravel()
    edges = np.arange(0.0, 0.42 + bin_width, bin_width)
    bc, bq, bv = [], [], []
    for k in range(len(edges) - 1):
        m = (oc >= edges[k]) & (oc < edges[k + 1])
        if int(m.sum()) >= min_cnt:
            bc.append(0.5 * (edges[k] + edges[k + 1]))
            bq.append(float(fl[m].mean()))
            bv.append(float(sp[m].mean()))
    return np.array(bc), np.array(bq), np.array(bv)


def plot_pems04_vs_pems08_thresholds(
    data08_tr: np.ndarray,
    th08: CongestionThresholds,
    data04_tr: np.ndarray,
    th04: CongestionThresholds,
    save_path: Path,
) -> None:
    """Hình 1 (1×2): So sánh Biểu đồ cơ bản Q(o) và V(o) thuần trên Train (60%) của PEMS08 vs. PEMS04."""
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2))
    bc8, bq8, bv8 = _bin_fd(data08_tr)
    bc4, bq4, bv4 = _bin_fd(data04_tr)

    ax = axes[0]
    ax.grid(True, alpha=0.65)
    ax.plot(bc8, bq8, color=PAL["navy"], lw=2.4, marker="o", ms=3.5, label=f"PEMS08 Train (o_sat = {th08.o_sat:.4f})")
    ax.plot(bc4, bq4, color=PAL["coral"], lw=2.4, ls="--", marker="s", ms=3.5, label=f"PEMS04 Train (o_sat = {th04.o_sat:.4f})")
    ax.axvline(th08.o_sat, color=PAL["navy"], ls=":", lw=1.4)
    ax.axvline(th04.o_sat, color=PAL["coral"], ls=":", lw=1.4)
    ax.annotate(
        f"PEMS04 tự hạ o_sat\n{th08.o_sat:.4f} → {th04.o_sat:.4f}",
        xy=(th04.o_sat, 415),
        xytext=(0.24, 510),
        fontsize=8.4,
        fontweight="bold",
        color=PAL["coral"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=PAL["coral"], lw=0.9),
        arrowprops=dict(arrowstyle="->", color=PAL["coral"], lw=1.2),
    )
    ax.set_xlim(0, 0.42)
    ax.set_ylim(0, 580)
    ax.set_xlabel("Độ chiếm dụng (Occupancy)")
    ax.set_ylabel("Lưu lượng TB Q(o) (xe / 5 phút)")
    ax.set_title("(A) Biểu đồ cơ bản Q(o) thuần Train: Tự hiệu chuẩn o_sat", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="lower right", fontsize=8.4)

    ax = axes[1]
    ax.grid(True, alpha=0.65)
    ax.plot(bc8, bv8, color=PAL["navy"], lw=2.4, marker="o", ms=3.5, label=f"PEMS08 Train (v_cong = {th08.v_cong:.2f} mph)")
    ax.plot(bc4, bv4, color=PAL["coral"], lw=2.4, ls="--", marker="s", ms=3.5, label=f"PEMS04 Train (v_cong = {th04.v_cong:.2f} mph)")
    ax.axhline(th08.v_cong, color=PAL["navy"], ls=":", lw=1.2, alpha=0.8)
    ax.axhline(th04.v_cong, color=PAL["coral"], ls="-.", lw=1.2, alpha=0.8)
    ax.axvline(th04.o_sat, color=PAL["coral"], ls=":", lw=1.3)
    ax.axvline(th08.o_sat, color=PAL["navy"], ls=":", lw=1.3)
    ax.text(
        0.39,
        th04.v_cong + 2.5,
        f"FHWA <60% FFS đồng nhất:\nPEMS08 = {th08.v_cong:.2f} | PEMS04 = {th04.v_cong:.2f} mph",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=PAL["slate"],
        bbox=dict(boxstyle="round,pad=0.28", fc="#F8FAFC", ec=PAL["grid"]),
    )
    ax.set_xlim(0, 0.42)
    ax.set_ylim(12, 74)
    ax.set_xlabel("Độ chiếm dụng (Occupancy)")
    ax.set_ylabel("Vận tốc TB V(o) (mph)")
    ax.set_title("(B) Đường V(o) & Ranh giới tắc nghẽn tự thích nghi giữa 2 bản đồ", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="upper right", fontsize=8.4)

    fig.tight_layout(w_pad=2.2)
    _save_and_show(fig, save_path)


def plot_pems04_ablation_mae(
    ablation04_df: pd.DataFrame,
    histories04: Dict[str, Dict[str, List[float]]],
    save_path: Path,
) -> None:
    """Hình 2 (2×2): Đối chứng 6 cấu hình đặc trưng huấn luyện độc lập trên PEMS04 (đối xứng 1-1 với NB04)."""
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.2))
    cfgs = ablation04_df["Cấu hình"].tolist()
    short_lbls = [c.replace("Base + Nhóm ", "+G").replace(" (", "\n(") for c in cfgs]
    x = np.arange(len(cfgs))
    bar_cols = ["#CBD5E1", PAL["purple"], PAL["blue"], PAL["teal"], PAL["amber"], PAL["navy"]]

    # (A) MAE Speed
    ax = axes[0, 0]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals = ablation04_df["MAE Speed (mph)"].to_numpy()
    bars = ax.bar(x, vals, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_v = vals[1]
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
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85),
        )
    ax.axhline(base_v, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ax.set_ylim(max(0.0, vals.min() * 0.88), vals.max() * 1.12)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("PEMS04 Test MAE Speed (mph) — TB 5–60p")
    ax.set_title("(A) Sai số Vận tốc toàn mạng trên PEMS04 (MAE Speed)", fontweight="bold")

    # (B) Normalized Std-MAE
    ax = axes[0, 1]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals_s = ablation04_df["Std-MAE (3 biến)"].to_numpy()
    bars_s = ax.bar(x, vals_s, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_s = vals_s[1]
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
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85),
        )
    ax.axhline(base_s, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ax.set_ylim(max(0.0, vals_s.min() * 0.88), vals_s.max() * 1.12)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("PEMS04 Test Std-MAE (trung bình chuẩn hóa 3 biến)")
    ax.set_title("(B) Sai số tổng hợp 3 biến trên PEMS04 (% so với Base)", fontweight="bold")

    # (C) Bottleneck stations Speed MAE
    ax = axes[1, 0]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    vals_c = ablation04_df["MAE Speed Điểm nghẽn (mph)"].to_numpy()
    bars_c = ax.bar(x, vals_c, width=0.60, color=bar_cols[: len(cfgs)], edgecolor="#64748B", lw=0.6)
    base_c = vals_c[1]
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
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85),
        )
    ax.axhline(base_c, color=PAL["purple"], ls="--", lw=1.0, alpha=0.75)
    ax.set_ylim(max(0.0, vals_c.min() * 0.88), vals_c.max() * 1.12)
    ax.set_xticks(x, short_lbls, fontsize=8.4)
    ax.set_ylabel("PEMS04 Test MAE Speed tại Điểm nghẽn (mph)")
    ax.set_title("(C) Sai số Vận tốc trên 155 trạm Điểm nghẽn PEMS04 (thuần Train)", fontweight="bold")

    # (D) Validation L1 learning curves
    ax = axes[1, 1]
    ax.grid(True, alpha=0.65)
    for cname, col, ls, mk in [
        ("Raw (3)", PAL["muted"], "--", "o"),
        ("Base (7)", PAL["purple"], "-.", "s"),
        ("Full (14)", PAL["navy"], "-", "^"),
    ]:
        if cname in histories04 and len(histories04[cname].get("val_loss", [])) > 0:
            vl = histories04[cname]["val_loss"]
            ep = np.arange(1, len(vl) + 1)
            ax.plot(ep, vl, color=col, ls=ls, lw=2.1, marker=mk, ms=4.0, label=f"{cname} (best={min(vl):.4f})")
    ax.set_xlabel("Epoch huấn luyện trên PEMS04")
    ax.set_ylabel("Validation L1 Loss (Std-MAE)")
    ax.set_title("(D) Quá trình hội tụ Validation L1 Loss trên PEMS04 (SEED = 42)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], fontsize=8.4)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    _save_and_show(fig, save_path)


def plot_cross_map_pfi_and_attention(
    grp08_df: pd.DataFrame,
    grp04_df: pd.DataFrame,
    attn04_res: Dict[str, object],
    save_path: Path,
) -> None:
    """Hình 3 (1×2): Đối chiếu PFI 4 nhóm đặc trưng giữa PEMS08 vs. PEMS04 & Chú ý GATv2 trên PEMS04."""
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4))

    ax = axes[0]
    ax.grid(axis="x", alpha=0.65)
    ax.set_axisbelow(True)
    order_grps = [
        "Nhóm 2 — Liên kết đa biến",
        "Nhóm 1 — Dao động khử dốc",
        "Nhóm 3 — Độ dốc có dấu",
        "Nhóm 4 — STID (Chu kỳ thời gian)",
    ]
    m08 = grp08_df.set_index("group")["delta_std_mae"].to_dict()
    m04 = grp04_df.set_index("group")["delta_std_mae"].to_dict()
    v08 = [float(m08.get(g, 0.0)) for g in order_grps]
    v04 = [float(m04.get(g, 0.0)) for g in order_grps]

    y = np.arange(len(order_grps))
    h = 0.35
    b8 = ax.barh(y + h / 2, v08, height=h, color=PAL["navy"], alpha=0.90, label="PEMS08 (170 trạm, NB04)")
    b4 = ax.barh(y - h / 2, v04, height=h, color=PAL["coral"], alpha=0.88, label="PEMS04 (307 trạm, NB05)")
    xmax = max(max(v08), max(v04))
    for bars, col in [(b8, PAL["navy"]), (b4, PAL["coral"])]:
        for bar in bars:
            w_val = bar.get_width()
            ax.text(w_val + xmax * 0.02, bar.get_y() + bar.get_height() / 2, f"+{w_val:.4f}", va="center", fontsize=8.1, fontweight="bold", color=col)
    ax.set_yticks(y, [g.replace(" — ", "\n") for g in order_grps], fontsize=8.8)
    ax.set_xlim(0, xmax * 1.28)
    ax.set_xlabel("Mức tăng Std-MAE khi xáo trộn nhóm (ΔStd-MAE)")
    ax.set_title("(A) Tính nhất quán của 4 nhóm đặc trưng: PEMS08 vs. PEMS04", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="lower right", fontsize=8.4)

    ax = axes[1]
    sm = attn04_res["summary_means"]
    cats = ["Self\n(Bản thân)", "In\n(Thượng nguồn)", "Out\n(Hạ nguồn dội ngược)"]
    norm_v = [sm["self_normal"], sm["in_normal"], sm["out_normal"]]
    cong_v = [sm["self_pre_cong"], sm["in_pre_cong"], sm["out_pre_cong"]]
    xc = np.arange(3)
    w = 0.34
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    b1 = ax.bar(xc - w / 2, norm_v, width=w, color="#CBD5E1", edgecolor="#94A3B8", lw=0.6, label="Bình thường (C = 0)")
    b2 = ax.bar(xc + w / 2, cong_v, width=w, color=PAL["coral"], alpha=0.90, label="Tiền tắc / Sắp tắc (C = 1)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.set_xticks(xc, cats)
    ax.set_ylim(0, max(norm_v + cong_v) * 1.24)
    ax.set_ylabel("Tổng trọng số chú ý cạnh TB (∑α_ij)")
    ax.set_title(f"(B) Phân bổ chú ý GATv2 trên PEMS04 ({len(attn04_res['bn_summary_df'])} trạm điểm nghẽn)", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=8.4)

    fig.tight_layout(w_pad=2.4)
    _save_and_show(fig, save_path)


def plot_final_cross_map_synthesis(
    ablation08_df: pd.DataFrame,
    ablation04_df: pd.DataFrame,
    horizon_curves04: Dict[str, Dict[str, np.ndarray]],
    cong04_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Hình 4 (1×3): Tổng kết đối chứng công bằng PEMS08 vs. PEMS04 (% giảm Std-MAE, tầm 5p→60p & F1 phụ)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.1))

    # (A) % Std-MAE reduction vs Raw (3) across PEMS08 vs PEMS04
    ax = axes[0]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    cfgs_sub = ablation08_df["Cấu hình"].tolist()[1:]  # Base(7), +G1, +G2, +G3, Full(14)
    short_c = [c.replace("Base + Nhóm ", "+G").replace(" (", "\n(") for c in cfgs_sub]
    r08 = float(ablation08_df.loc[ablation08_df["Cấu hình"] == "Raw (3)", "Std-MAE (3 biến)"].iloc[0])
    r04 = float(ablation04_df.loc[ablation04_df["Cấu hình"] == "Raw (3)", "Std-MAE (3 biến)"].iloc[0])
    g08 = [-(float(ablation08_df.loc[ablation08_df["Cấu hình"] == c, "Std-MAE (3 biến)"].iloc[0]) - r08) / r08 * 100.0 for c in cfgs_sub]
    g04 = [-(float(ablation04_df.loc[ablation04_df["Cấu hình"] == c, "Std-MAE (3 biến)"].iloc[0]) - r04) / r04 * 100.0 for c in cfgs_sub]

    x = np.arange(len(cfgs_sub))
    w = 0.36
    b8 = ax.bar(x - w / 2, g08, width=w, color=PAL["navy"], alpha=0.90, label="PEMS08 (170 trạm)")
    b4 = ax.bar(x + w / 2, g04, width=w, color=PAL["coral"], alpha=0.88, label="PEMS04 (307 trạm)")
    ymax_g = max(max(g08), max(g04))
    for i, (bar8, bar4) in enumerate(zip(b8, b4)):
        fw = "bold" if i == len(cfgs_sub) - 1 else "normal"
        ax.text(bar8.get_x() + bar8.get_width() / 2, bar8.get_height() + ymax_g * 0.025, f"-{bar8.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight=fw, color=PAL["navy"])
        ax.text(bar4.get_x() + bar4.get_width() / 2, bar4.get_height() + ymax_g * 0.025, f"-{bar4.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight=fw, color=PAL["coral"])
    ax.set_xticks(x, short_c, fontsize=8.0)
    ax.set_ylim(0, ymax_g * 1.24)
    ax.set_ylabel("Mức giảm Test Std-MAE so với Raw (3) (%)")
    ax.set_title("(A) Mức giảm Std-MAE lặp lại trên cả 2 thành phố", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=8.2)

    # (B) PEMS04 MAE Speed across 5..60m: Raw(3) vs Base(7) vs Full(14) (+ Zero-Shot Full 14)
    ax = axes[1]
    ax.grid(True, alpha=0.65)
    steps_min = np.arange(1, 13) * 5
    for cname, col, ls, mk in [
        ("Zero-Shot Full (14)", PAL["amber"], ":", "d"),
        ("Raw (3)", PAL["muted"], "--", "o"),
        ("Base (7)", PAL["purple"], "-.", "s"),
        ("Full (14)", PAL["navy"], "-", "^"),
    ]:
        if cname in horizon_curves04:
            ax.plot(steps_min, horizon_curves04[cname]["mae_speed"], color=col, ls=ls, lw=2.3 if cname == "Full (14)" else 1.9, marker=mk, ms=4.0, label=cname)
    ax.set_xticks([5, 15, 30, 45, 60])
    ax.set_xlabel("Tầm dự báo tương lai Δt (phút)")
    ax.set_ylabel("PEMS04 Test MAE Speed (mph)")
    ax.set_title("(B) MAE Vận tốc PEMS04 qua 12 tầm dự báo", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="upper left", fontsize=7.9)

    # (C) Secondary congestion marking F1 on PEMS04 (Raw 3 vs Base 7 vs Full 14)
    ax = axes[2]
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)
    xh = np.arange(len(cong04_df))
    w3 = 0.25
    for idx_c, (cname, col) in enumerate([("Raw (3)", "#CBD5E1"), ("Base (7)", PAL["purple"]), ("Full (14)", PAL["navy"])]):
        vs = cong04_df[f"F1 {cname}"].to_numpy()
        bars = ax.bar(xh + (idx_c - 1) * w3, vs, width=w3 * 0.94, color=col, edgecolor="#64748B", lw=0.5, label=cname)
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold" if idx_c == 2 else "normal")
    ax.set_xticks(xh, cong04_df["Tầm dự báo"].tolist())
    vmax_f1 = cong04_df[[c for c in cong04_df.columns if c.startswith("F1 ")]].to_numpy().max()
    ax.set_ylim(0, min(1.05, vmax_f1 * 1.22))
    ax.set_ylabel("F1-Score đánh dấu tắc (ngưỡng Train PEMS04)")
    ax.set_title("(C) Đánh giá phụ: Đánh dấu tắc nghẽn PEMS04", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=8.1)

    fig.tight_layout(w_pad=2.0)
    _save_and_show(fig, save_path)
