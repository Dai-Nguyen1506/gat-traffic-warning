"""Publication-grade plotting utilities for NB03 (RQ1 on PEMS08)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.bottlenecks import CSD_FEATURE_NAMES, CongestionThresholds, merge_phase_spans

PAL = {
    "navy": "#1B365D",
    "blue": "#2B6CB0",
    "coral": "#D64545",
    "amber": "#E69F00",
    "teal": "#1E8E7E",
    "slate": "#475569",
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "savefig.dpi": 300,
        "figure.dpi": 115,
        "font.size": 10,
    })


def _save_and_show(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu biểu đồ: {save_path.name}")


def plot_pems08_overview(
    flow: np.ndarray,
    occ: np.ndarray,
    spd: np.ndarray,
    slow10: Sequence[int],
    mat_spd: np.ndarray,
    corr_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Hình 1: 3 chuỗi 14 ngày (Speed, Occupancy, Flow) + 2 nhiệt đồ (Thứ×Giờ & Tương quan 3×3)."""
    fig = plt.figure(figsize=(15.2, 10.0))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.28, 1.0], hspace=0.34)
    top = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.14)
    bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.65, 1.0], wspace=0.22)
    ax_sp, ax_oc, ax_fl = fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(top[2])
    ax_hm, ax_corr = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])

    n_days = 14
    days_x = np.arange(n_days * 288) / 288.0
    sp_s = pd.Series(spd[: n_days * 288, slow10].mean(axis=1)).rolling(6, min_periods=1, center=True).mean()
    oc_s = pd.Series(occ[: n_days * 288, slow10].mean(axis=1)).rolling(6, min_periods=1, center=True).mean()
    fl_s = pd.Series(flow[: n_days * 288, slow10].mean(axis=1)).rolling(6, min_periods=1, center=True).mean()

    for ax_t in (ax_sp, ax_oc, ax_fl):
        ax_t.grid(axis="y", alpha=0.72)
        ax_t.set_xlim(0, n_days)
        for d in range(n_days):
            if ((d + 4) % 7) >= 5:
                ax_t.axvspan(d, d + 1, color="#E2E8F0", alpha=0.65, lw=0)

    ax_sp.plot(days_x, sp_s, color=PAL["navy"], lw=1.7)
    ax_sp.fill_between(days_x, 28, sp_s, color=PAL["blue"], alpha=0.10)
    ax_sp.set_ylim(28, 73); ax_sp.set_ylabel("Vận tốc TB\n(mph)")
    ax_sp.set_title("(A) Diễn biến 14 ngày đầu của đủ 3 biến — Top 10 trạm chịu tải nặng nhất", fontweight="bold", pad=8)
    for x_wk in (2.0, 9.0):
        ax_sp.text(x_wk, 71.5, "Cuối tuần", ha="center", va="top", fontsize=8.3, color=PAL["slate"], fontweight="bold")
    plt.setp(ax_sp.get_xticklabels(), visible=False)

    ax_oc.plot(days_x, oc_s, color=PAL["coral"], lw=1.7)
    ax_oc.fill_between(days_x, 0, oc_s, color=PAL["coral"], alpha=0.12)
    ax_oc.set_ylim(0, 0.28); ax_oc.set_ylabel("Độ chiếm dụng\n(Occupancy)")
    plt.setp(ax_oc.get_xticklabels(), visible=False)

    ax_fl.plot(days_x, fl_s, color=PAL["amber"], lw=1.7)
    ax_fl.fill_between(days_x, 0, fl_s, color=PAL["amber"], alpha=0.15)
    ax_fl.set_ylim(0, 480); ax_fl.set_xticks(range(0, n_days + 1, 1))
    ax_fl.set_ylabel("Lưu lượng TB\n(xe / 5 phút)"); ax_fl.set_xlabel("Ngày kể từ Thứ Sáu 01/07/2016 (vùng xám = Thứ 7 & Chủ nhật)")

    im = ax_hm.imshow(mat_spd, aspect="auto", cmap="RdYlBu", vmin=34, vmax=68)
    ax_hm.set_yticks(range(7), ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"])
    ax_hm.set_xticks(range(0, 24, 2), [f"{h:02d}h" for h in range(0, 24, 2)])
    ax_hm.set_xticks(np.arange(-0.5, 24, 1), minor=True); ax_hm.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax_hm.grid(which="minor", color="white", linestyle="-", linewidth=1.0); ax_hm.tick_params(which="minor", bottom=False, left=False)
    for s in ax_hm.spines.values():
        s.set_visible(False)
    ax_hm.set_xlabel("Giờ trong ngày"); ax_hm.set_title("(B) Nhiệt đồ Thứ × Giờ — Vận tốc TB Top 10 trạm nặng tải (mph)", fontweight="bold", pad=10)
    cbar = plt.colorbar(im, ax=ax_hm, fraction=0.036, pad=0.025); cbar.outline.set_visible(False); cbar.set_label("Vận tốc TB (mph)")

    cmat = corr_df.to_numpy()
    im_c = ax_corr.imshow(cmat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    lbls_c = ["Flow\n(Lưu lượng)", "Occupancy\n(Chiếm dụng)", "Speed\n(Vận tốc)"]
    ax_corr.set_xticks(range(3), lbls_c, fontsize=9.2); ax_corr.set_yticks(range(3), lbls_c, fontsize=9.2)
    ax_corr.set_xticks(np.arange(-0.5, 3, 1), minor=True); ax_corr.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_corr.grid(which="minor", color="white", linestyle="-", linewidth=2.0); ax_corr.tick_params(which="minor", bottom=False, left=False)
    for s in ax_corr.spines.values():
        s.set_visible(False)
    for i_r in range(3):
        for j_c in range(3):
            val = cmat[i_r, j_c]
            ax_corr.text(j_c, i_r, f"{val:+.2f}", ha="center", va="center", fontsize=11.5, fontweight="bold", color="white" if abs(val) >= 0.55 else "#0F172A")
    ax_corr.set_title("(C) Ma trận tương quan Pearson (3 × 3)", fontweight="bold", pad=10)
    cbar_c = plt.colorbar(im_c, ax=ax_corr, fraction=0.046, pad=0.04); cbar_c.outline.set_visible(False); cbar_c.set_label("Hệ số r")

    _save_and_show(fig, save_path)


def plot_fundamental_and_workday(
    flow: np.ndarray,
    occ: np.ndarray,
    spd: np.ndarray,
    cg: np.ndarray,
    th: CongestionThresholds,
    r_occ_spd: float,
    cap_drop_pct: float,
    n_empty_slow: int,
    save_path: Path,
    seed: int = 42,
) -> None:
    """Hình 2 (Thuần Phần 2): Biểu đồ cơ bản Q(o), đường giảm nghịch V(o) + vùng Tắc thực, và ngày thường N22."""
    fig = plt.figure(figsize=(14.6, 8.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 0.92], hspace=0.34, wspace=0.22)
    ax_fd, ax_sp, ax_wd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    rng = np.random.default_rng(seed)
    fl_r, oc_r, sp_r, cg_r = flow.ravel(), occ.ravel(), spd.ravel(), cg.ravel()
    samp = rng.choice(len(fl_r), size=35000, replace=False)

    ob = np.linspace(0.0, 0.42, 43)
    bc, bq, bv = [], [], []
    for k in range(42):
        mk = (oc_r >= ob[k]) & (oc_r < ob[k + 1])
        if mk.sum() >= 250:
            bc.append(0.5 * (ob[k] + ob[k + 1]))
            bq.append(float(fl_r[mk].mean()))
            bv.append(float(sp_r[mk].mean()))

    sc = ax_fd.scatter(oc_r[samp], fl_r[samp], c=sp_r[samp], cmap="RdYlBu", vmin=18, vmax=68, s=4, alpha=0.32, rasterized=True)
    ax_fd.plot(bc, bq, color=PAL["navy"], lw=2.5, label="Lưu lượng TB Q(o)")
    ax_fd.axvline(th.o_sat, color="#0F172A", ls="--", lw=1.3, label=f"o_sat = {th.o_sat:.4f}")
    ax_fd.axvline(th.o_crit, color=PAL["coral"], ls=":", lw=1.5, label=f"o_crit = {th.o_crit:.4f}")
    ax_fd.annotate(f"Capacity Drop ({cap_drop_pct:.1f}%)\nFlow tụt khi quá tải", xy=(0.34, 388), xytext=(0.355, 565), fontsize=8.3, fontweight="bold", color=PAL["coral"], ha="center", bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["coral"], lw=0.9), arrowprops=dict(arrowstyle="->", color=PAL["coral"], lw=1.2))
    ax_fd.grid(True, alpha=0.6); ax_fd.set_xlim(0, 0.48); ax_fd.set_ylim(0, 880)
    ax_fd.set_xlabel("Độ chiếm dụng (Occupancy)"); ax_fd.set_ylabel("Lưu lượng (Flow, xe / 5p)")
    ax_fd.set_title("(A) Biểu đồ cơ bản PEMS08: Đỉnh công suất & Capacity Drop", fontweight="bold")
    ax_fd.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor=PAL["grid"], loc="upper right", fontsize=8.1)
    cbar = plt.colorbar(sc, ax=ax_fd, pad=0.02, fraction=0.045); cbar.outline.set_visible(False); cbar.set_label("Vận tốc (mph)")

    oc_s, sp_s, cg_s = oc_r[samp], sp_r[samp], cg_r[samp]
    m_empty = (sp_s < th.v_cong) & (oc_s < th.o_sat)
    m_free = (~cg_s) & (~m_empty)
    ax_sp.scatter(oc_s[m_free], sp_s[m_free], s=4, alpha=0.24, color="#94A3B8", label="Thông thoáng (C = 0)", rasterized=True)
    ax_sp.scatter(oc_s[cg_s], sp_s[cg_s], s=6, alpha=0.48, color=PAL["coral"], label=f"Tắc thực: Speed < {th.v_cong:.1f} & Occ ≥ {th.o_sat:.4f}", rasterized=True)
    ax_sp.scatter(oc_s[m_empty], sp_s[m_empty], s=12, alpha=0.88, color="#334155", marker="x", label=f"Chậm nhưng đường vắng (loại: {n_empty_slow:,} điểm)", rasterized=True)
    ax_sp.plot(bc, bv, color=PAL["navy"], lw=2.5, zorder=5, label=f"Vận tốc TB V(o) (r = {r_occ_spd:.2f})")
    ax_sp.axhline(th.v_cong, color=PAL["coral"], ls="--", lw=1.35)
    ax_sp.axvline(th.o_sat, color="#0F172A", ls=":", lw=1.35)
    ax_sp.grid(True, alpha=0.6); ax_sp.set_xlim(0, 0.45); ax_sp.set_ylim(5, 78)
    ax_sp.set_xlabel("Độ chiếm dụng (Occupancy)"); ax_sp.set_ylabel("Vận tốc (mph)")
    ax_sp.set_title("(B) Đường V(o) giảm nghịch & vùng Tắc thực (Speed – Occ)", fontweight="bold")
    ax_sp.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor=PAL["grid"], loc="upper right", fontsize=7.8, markerscale=1.4)

    s0, s1 = 6 * 288, 7 * 288
    hrs_24 = np.arange(288) * 5.0 / 60.0
    for i_s, (a, b) in enumerate(merge_phase_spans(hrs_24, cg[s0:s1, 22], 2.5 / 60)):
        ax_wd.axvspan(a, b, color=PAL["coral"], alpha=0.18, lw=0, label="Khung giờ Tắc thực (C = 1)" if i_s == 0 else None)
    ax_wd.plot(hrs_24, spd[s0:s1, 22], color=PAL["navy"], lw=2.0, label="Vận tốc thực Speed (mph)")
    ax_wd.axhline(th.v_cong, color=PAL["coral"], ls="--", lw=1.3, label=f"Ngưỡng tắc tự tính v_cong = {th.v_cong:.1f} mph")
    ax_wd.grid(axis="y", alpha=0.7); ax_wd.set_xlim(0, 24); ax_wd.set_ylim(14, 84)
    ax_wd.set_xticks(range(0, 25, 2), [f"{h:02d}:00" for h in range(0, 25, 2)])
    ax_wd.set_xlabel("Khung giờ trong ngày làm việc (Thứ Năm 07/07/2016 — Trạm N22)"); ax_wd.set_ylabel("Vận tốc (mph)")
    ax_wd.set_title("(C) Áp dụng điều kiện tắc tự tính lên 1 ngày thường thực tế: Xác nhận khung giờ tắc", fontweight="bold")
    ax_wd.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.4)

    _save_and_show(fig, save_path)


def plot_precongestion_trajectories(
    rel_steps: np.ndarray,
    curves: dict,
    th: CongestionThresholds,
    save_path: Path,
) -> None:
    """Hình 3a (Lưới 2 × 2): Quỹ đạo trung bình trọn chu kỳ (-60p → +90p) của Biến gốc và đủ 7 chỉ báo."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8))
    mins = rel_steps * 5
    spd_c, occ_c, sl_c = np.array(curves["spd"]), np.array(curves["occ"]), np.array(curves["spd_slope"])
    pre_spans = merge_phase_spans(mins, ((spd_c < th.v_sync) | (occ_c >= th.o_sat)) & (mins < 0), 2.5)
    jam_spans = merge_phase_spans(mins, (spd_c < th.v_cong) & (mins >= 0), 2.5)
    rec_spans = merge_phase_spans(mins, (sl_c > 0) & (mins > 0), 2.5)

    def _prep(ax_t):
        ax_t.grid(axis="y", alpha=0.65); ax_t.set_axisbelow(True)
        for a, b in pre_spans:
            ax_t.axvspan(a, b, color=PAL["amber"], alpha=0.14, lw=0)
        for a, b in jam_spans:
            ax_t.axvspan(a, b, color=PAL["coral"], alpha=0.10, lw=0)
        for a, b in rec_spans:
            ax_t.axvspan(a, b, color=PAL["teal"], alpha=0.09, lw=0)
        ax_t.axvline(0, color=PAL["coral"], ls="--", lw=1.35)
        ax_t.set_xlim(-62, 92); ax_t.set_xticks([-60, -30, 0, 30, 60, 90])

    ax = axes[0, 0]
    _prep(ax)
    l1 = ax.plot(mins, spd_c, color=PAL["navy"], lw=2.3, marker="o", ms=3.0, label="Vận tốc TB (mph)")
    ax.axhline(th.v_cong, color=PAL["navy"], ls=":", lw=1.0, alpha=0.6)
    ax.set_ylabel("Vận tốc (mph)", color=PAL["navy"]); ax.set_xlabel("Thời gian so với t = 0 (phút)  [vàng: chuyển tiếp | đỏ: tắc | xanh: hồi phục]")
    ax2 = ax.twinx()
    l2 = ax2.plot(mins, occ_c, color=PAL["coral"], lw=2.1, ls="-.", marker="s", ms=2.8, label="Độ chiếm dụng TB (Occupancy)")
    ax2.set_ylabel("Occupancy", color=PAL["coral"]); ax2.spines["top"].set_visible(False)
    ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="center left", fontsize=8.3)
    ax.set_title("(A) Biến gốc trọn chu kỳ: Chuyển pha → Sụp tắc → Hồi phục", fontweight="bold")

    ax = axes[0, 1]
    _prep(ax)
    for fn, lbl, col, mk in [("occ_rstd", "occ_rstd (Dao động Occ)", PAL["navy"], "o"), ("spd_rstd", "spd_rstd (Dao động Speed)", PAL["amber"], "s"), ("flow_rstd", "flow_rstd (Dao động Flow)", PAL["muted"], "^")]:
        arr = np.array(curves[fn])
        ax.plot(mins, arr / arr[0], color=col, lw=2.2, marker=mk, ms=3.0, label=lbl)
    ax.axhline(1.0, color=PAL["muted"], ls=":", lw=1.0)
    ax.set_ylabel("Bội số so với mức nền tại -60p (×)"); ax.set_xlabel("Thời gian so với khởi phát tắc t = 0 (phút)")
    ax.set_title("(B) Nhóm 1: Ba chỉ báo Dao động khử dốc (Kerner, 2004)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="upper left", fontsize=8.3)

    ax = axes[1, 0]
    _prep(ax)
    ax.plot(mins, curves["occ_spd_corr"], color=PAL["teal"], lw=2.3, marker="s", ms=3.0, label="occ_spd_corr (-corr(Occ, Speed))")
    ax.plot(mins, curves["flow_occ_decouple"], color=PAL["navy"], lw=2.2, ls="-.", marker="o", ms=2.8, label="flow_occ_decouple (-corr(Flow, Occ))")
    ax.axhline(0.0, color=PAL["slate"], ls=":", lw=1.0)
    ax.set_ylim(-0.95, 1.15)
    ax.set_ylabel("Hệ số liên kết đa biến [-1, +1]"); ax.set_xlabel("Thời gian so với khởi phát tắc t = 0 (phút)")
    ax.set_title("(C) Nhóm 2: Hai chỉ báo Liên kết Đa biến (Chattopadhyay & Gupta, 2026)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="lower right", fontsize=8.2)

    ax = axes[1, 1]
    _prep(ax)
    l1 = ax.plot(mins, sl_c, color=PAL["coral"], lw=2.3, marker="o", ms=3.0, label="spd_slope (< 0 tụt tốc; > 0 hồi phục)")
    ax.axhline(0, color=PAL["slate"], ls=":", lw=1.1)
    ax.set_ylim(-2.8, 1.35)
    ax.set_ylabel("spd_slope (mph / 5p)", color=PAL["coral"]); ax.set_xlabel("Thời gian so với khởi phát tắc t = 0 (phút)")
    ax2 = ax.twinx()
    l2 = ax2.plot(mins, curves["occ_slope"], color=PAL["blue"], lw=2.2, ls="-.", marker="s", ms=2.8, label="occ_slope (> 0 dồn ứ; < 0 đang thoát)")
    ax2.set_ylim(-0.006, 0.013)
    ax2.set_ylabel("occ_slope (/ 5p)", color=PAL["blue"]); ax2.spines["top"].set_visible(False)
    ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], frameon=True, facecolor="white", edgecolor=PAL["grid"], loc="lower left", fontsize=8.2)
    ax.set_title("(D) Nhóm 3: Hai chỉ báo Độ dốc có dấu (chống báo giả hồi phục)", fontweight="bold")

    fig.tight_layout(w_pad=2.5, h_pad=2.6)
    _save_and_show(fig, save_path)


def plot_stid_time_cycles(
    cg: np.ndarray,
    hr_all: np.ndarray,
    dow_all: np.ndarray,
    time4: np.ndarray,
    save_path: Path,
) -> None:
    """Hình 3b (2 × 2 dùng chung trục x): Trực quan hóa Nhóm 4 (STID — tod_sin/cos & dow_sin/cos) đối chiếu P(C=1) toàn mạng."""
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(15.2, 6.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.28, 0.92], hspace=0.14, wspace=0.22)
    ax_h_top = fig.add_subplot(gs[0, 0])
    ax_h_bot = fig.add_subplot(gs[1, 0], sharex=ax_h_top)
    ax_d_top = fig.add_subplot(gs[0, 1])
    ax_d_bot = fig.add_subplot(gs[1, 1], sharex=ax_d_top)

    hr_rate = np.array([cg[hr_all == h].mean() * 100.0 for h in range(24)])
    tod_sin_h = np.array([time4[hr_all == h, 0, 0].mean() for h in range(24)])
    tod_cos_h = np.array([time4[hr_all == h, 0, 1].mean() for h in range(24)])

    dow_rate = np.array([cg[dow_all == d].mean() * 100.0 for d in range(7)])
    dow_sin_d = np.array([time4[dow_all == d, 0, 2].mean() for d in range(7)])
    dow_cos_d = np.array([time4[dow_all == d, 0, 3].mean() for d in range(7)])

    h_am = int(np.argmax(hr_rate[6:11]) + 6)
    h_pm = int(np.argmax(hr_rate[12:20]) + 12)
    d_pk = int(np.argmax(dow_rate))

    # --- Panel (A): 24h cycle ---
    hrs = np.arange(24)
    ax_h_top.grid(axis="y", alpha=0.65); ax_h_top.set_axisbelow(True)
    bar_cols_h = [PAL["coral"] if (h in range(7, 10) or h in range(14, 19)) else "#94A3B8" for h in hrs]
    ax_h_top.bar(hrs, hr_rate, width=0.72, color=bar_cols_h, alpha=0.86, edgecolor="white", lw=0.6)
    for hx in (h_am, h_pm):
        ax_h_top.axvline(hx, color=PAL["coral"], ls=":", lw=1.1, alpha=0.75)
        ax_h_top.text(hx, hr_rate[hx] + 0.16, f"{hr_rate[hx]:.2f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=PAL["coral"])
    night_mean = float(hr_rate[:5].mean())
    ax_h_top.annotate(
        f"Đêm 00–04h:\nTB {night_mean:.2f}%",
        xy=(2.0, max(hr_rate[2], 0.08)),
        xytext=(2.3, hr_rate.max() * 0.56),
        ha="center",
        fontsize=8.2,
        color=PAL["slate"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec=PAL["grid"]),
        arrowprops=dict(arrowstyle="->", color=PAL["slate"], lw=0.9),
    )
    ax_h_top.set_ylim(0, hr_rate.max() * 1.24)
    ax_h_top.set_ylabel("Xác suất Tắc thực P(C=1) (%)")
    ax_h_top.set_title("(A) Chu kỳ 24 giờ: P(C = 1) toàn mạng & tod_sin / tod_cos", fontweight="bold", fontsize=10.6, pad=8)
    ax_h_top.legend(
        handles=[Patch(facecolor=PAL["coral"], alpha=0.86, label="Cao điểm (07–09h, 14–18h)"), Patch(facecolor="#94A3B8", alpha=0.86, label="Khung giờ còn lại")],
        frameon=False, loc="upper left", fontsize=8.0,
    )
    plt.setp(ax_h_top.get_xticklabels(), visible=False)

    ax_h_bot.grid(axis="y", alpha=0.65); ax_h_bot.set_axisbelow(True)
    ax_h_bot.axhline(0.0, color=PAL["muted"], ls=":", lw=0.9)
    for hx in (h_am, h_pm):
        ax_h_bot.axvline(hx, color=PAL["coral"], ls=":", lw=1.1, alpha=0.75)
    ax_h_bot.plot(hrs, tod_sin_h, color=PAL["navy"], lw=2.2, marker="o", ms=3.2, label="tod_sin (sin 2πt / 24h)")
    ax_h_bot.plot(hrs, tod_cos_h, color=PAL["teal"], lw=2.1, ls="--", marker="s", ms=3.0, label="tod_cos (cos 2πt / 24h)")
    ax_h_bot.set_ylim(-1.22, 1.48); ax_h_bot.set_yticks([-1.0, 0.0, 1.0])
    ax_h_bot.set_xlim(-0.7, 23.7); ax_h_bot.set_xticks(range(0, 24, 2), [f"{h:02d}h" for h in range(0, 24, 2)])
    ax_h_bot.set_ylabel("STID [-1, +1]"); ax_h_bot.set_xlabel("Giờ trong ngày (khép kín chu kỳ 23h ↔ 00h không đứt gãy)")
    ax_h_bot.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor=PAL["grid"], loc="upper center", ncol=2, fontsize=8.2)

    # --- Panel (B): 7-day cycle ---
    days = np.arange(7)
    day_lbls = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    ax_d_top.grid(axis="y", alpha=0.65); ax_d_top.set_axisbelow(True)
    bar_cols_d = [PAL["coral"] if d < 5 else "#94A3B8" for d in days]
    ax_d_top.bar(days, dow_rate, width=0.56, color=bar_cols_d, alpha=0.86, edgecolor="white", lw=0.7)
    ax_d_top.axvline(d_pk, color=PAL["coral"], ls=":", lw=1.1, alpha=0.75)
    for d_i, val in enumerate(dow_rate):
        ax_d_top.text(d_i, val + 0.08, f"{val:.2f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=PAL["coral"] if d_i < 5 else PAL["slate"])
    ax_d_top.set_ylim(0, dow_rate.max() * 1.26)
    ax_d_top.set_ylabel("Xác suất Tắc thực P(C=1) (%)")
    ax_d_top.set_title("(B) Chu kỳ 7 ngày: P(C = 1) theo thứ & dow_sin / dow_cos", fontweight="bold", fontsize=10.6, pad=8)
    ax_d_top.legend(
        handles=[Patch(facecolor=PAL["coral"], alpha=0.86, label="Ngày thường (T2–T6)"), Patch(facecolor="#94A3B8", alpha=0.86, label="Cuối tuần (T7–CN)")],
        frameon=False, loc="upper right", fontsize=8.0,
    )
    plt.setp(ax_d_top.get_xticklabels(), visible=False)

    ax_d_bot.grid(axis="y", alpha=0.65); ax_d_bot.set_axisbelow(True)
    ax_d_bot.axhline(0.0, color=PAL["muted"], ls=":", lw=0.9)
    ax_d_bot.axvline(d_pk, color=PAL["coral"], ls=":", lw=1.1, alpha=0.75)
    ax_d_bot.plot(days, dow_sin_d, color=PAL["navy"], lw=2.2, marker="o", ms=4.5, label="dow_sin (sin 2πd / 7)")
    ax_d_bot.plot(days, dow_cos_d, color=PAL["teal"], lw=2.1, ls="--", marker="s", ms=4.2, label="dow_cos (cos 2πd / 7)")
    ax_d_bot.set_ylim(-1.22, 1.48); ax_d_bot.set_yticks([-1.0, 0.0, 1.0])
    ax_d_bot.set_xlim(-0.55, 6.55); ax_d_bot.set_xticks(days, day_lbls, fontsize=8.8)
    ax_d_bot.set_ylabel("STID [-1, +1]"); ax_d_bot.set_xlabel("Thứ trong tuần (khép kín Chủ nhật ↔ Thứ 2, tách hẳn cuối tuần)")
    ax_d_bot.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor=PAL["grid"], loc="upper center", ncol=2, fontsize=8.2)

    _save_and_show(fig, save_path)


def plot_precongestion_validation(
    csd_df: pd.DataFrame,
    lrt_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Hình 3c (1 × 2): So sánh ROC-AUC của đủ 4 nhóm (9 mục) & Đóng góp từng nhóm + gộp đủ 14 đặc trưng (PR-AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4), gridspec_kw={"width_ratios": [1.22, 1.15]})

    # --- Panel (A): All 4 groups (9 items) side-by-side ---
    ax = axes[0]
    ax.grid(axis="y", alpha=0.65); ax.set_axisbelow(True)
    n_items = len(csd_df)
    x, w = np.arange(n_items), 0.36
    b1 = ax.bar(x - w / 2, csd_df["AUC (Thông sạch 60p)"], width=w, color=PAL["navy"], label="Toàn bộ điểm thông sạch 60p")
    b2 = ax.bar(x + w / 2, csd_df["AUC (Speed ≥ 55 mph)"], width=w, color=PAL["teal"], label="Đang chạy nhanh (Speed ≥ 55 mph)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.011, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.axhline(0.5, color=PAL["muted"], ls="--", lw=1.0)

    # Subtle group dividers and labels for Groups 1–4
    for sep in (2.5, 4.5, 6.5):
        ax.axvline(sep, color="#CBD5E1", ls="--", lw=1.0)
    for cx, glbl in [(1.0, "Nhóm 1\nDao động"), (3.5, "Nhóm 2\nLiên kết"), (5.5, "Nhóm 3\nĐộ dốc"), (7.5, "Nhóm 4\nSTID")]:
        ax.text(cx, 1.035, glbl, ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=PAL["slate"])

    x_lbls = [
        "occ\nrstd", "spd\nrstd", "flow\nrstd",
        "occ_spd\ncorr", "flow_occ\ndecouple",
        "occ\nslope", "spd\nslope",
        "tod\n(sin,cos)", "dow\n(sin,cos)",
    ]
    ax.set_xticks(x, x_lbls, fontsize=7.9)
    ax.set_ylim(0.4, 1.19); ax.set_ylabel("ROC-AUC (Δt = 15 phút)")
    ax.set_title("(A) Sức mạnh phân tách (ROC-AUC) của đủ 4 nhóm đặc trưng", fontweight="bold", fontsize=10.5, pad=8)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.99), fontsize=8.0)

    # --- Panel (B): Raw(3) vs each of the 4 groups individually vs Combined All 4 (14) ---
    ax = axes[1]
    ax.grid(axis="y", alpha=0.65); ax.set_axisbelow(True)
    x2 = np.arange(len(lrt_df))
    w2 = 0.132
    series_cfg = [
        ("Raw 3 biến", "#CBD5E1", "Raw 3 biến"),
        ("+Nhóm 1 (Dao động)", PAL["blue"], "+ Nhóm 1 (Dao động)"),
        ("+Nhóm 2 (Liên kết)", PAL["teal"], "+ Nhóm 2 (Liên kết)"),
        ("+Nhóm 3 (Độ dốc)", PAL["amber"], "+ Nhóm 3 (Độ dốc)"),
        ("+Nhóm 4 (STID)", "#7C6BA8", "+ Nhóm 4 (STID)"),
        ("Gộp 4 nhóm (14)", PAL["navy"], "Gộp cả 4 nhóm (14)"),
    ]
    offsets = (np.arange(len(series_cfg)) - (len(series_cfg) - 1) / 2.0) * w2
    for idx_s, (col_name, color, lbl) in enumerate(series_cfg):
        vals = lrt_df[col_name].to_numpy()
        edge = "#94A3B8" if idx_s == 0 else "white"
        ax.bar(x2 + offsets[idx_s], vals, width=w2 * 0.94, color=color, edgecolor=edge, lw=0.5, label=lbl)
        if idx_s == 0:
            for k in range(len(x2)):
                ax.text(x2[k] + offsets[idx_s], vals[k] + 0.005, f"{vals[k]:.3f}", ha="center", va="bottom", fontsize=7.2, color=PAL["slate"])
        elif idx_s == len(series_cfg) - 1:
            for k in range(len(x2)):
                ax.text(
                    x2[k] + offsets[idx_s],
                    vals[k] + 0.005,
                    f"{vals[k]:.3f}\n({lrt_df['Tăng PR-AUC (%)'].iloc[k]})",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color=PAL["navy"],
                )

    ax.set_xticks(x2, [f"Δt = {l}\n(LRT p = {p})" for l, p in zip(lrt_df["Tầm cảnh báo (Δt)"], lrt_df["p (LRT df=11)"])], fontsize=8.2)
    ax.set_ylim(0, 0.355); ax.set_ylabel("Validation PR-AUC (chia 60/20/20)")
    ax.set_title("(B) Đóng góp từng nhóm & Tổ hợp gộp đủ 4 nhóm (14 đặc trưng)", fontweight="bold", fontsize=10.5, pad=8)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    _save_and_show(fig, save_path)


def plot_spatial_and_graph(
    G: nx.DiGraph,
    st_df: pd.DataFrame,
    graph_lrt_df: pd.DataFrame,
    rho_b: float,
    p_b: float,
    save_path: Path,
) -> None:
    """Hình 4 (2 tầng): Bản đồ có hướng Kamada-Kawai + Đối chiếu Đồ thị Tĩnh (Betweenness) vs. Đồ thị Động (In/Out)."""
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.22)
    ax_g, ax_b, ax_d = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    df_idx = st_df.set_index("node_id")
    pos_raw = nx.kamada_kawai_layout(G.to_undirected())
    pos = {n: (xy[0] * 1.65, xy[1]) for n, xy in pos_raw.items()}
    hot_set = set(st_df.loc[st_df["congestion_freq"] >= 0.01, "node_id"].tolist())
    quiet_edges = [(u, v) for u, v in G.edges() if (u not in hot_set) and (v not in hot_set)]
    hot_edges = [(u, v) for u, v in G.edges() if (u in hot_set) or (v in hot_set)]

    nx.draw_networkx_edges(G, pos, edgelist=quiet_edges, ax=ax_g, alpha=0.38, edge_color="#CBD5E1", arrows=True, arrowsize=6.5, width=0.85, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=hot_edges, ax=ax_g, alpha=0.72, edge_color="#64748B", arrows=True, arrowsize=8.5, width=1.25, connectionstyle="arc3,rad=0.05")

    quiet_nodes = [n for n in G.nodes() if n not in hot_set]
    hot_nodes = sorted(hot_set, key=lambda u: df_idx.loc[u, "congestion_freq"])
    nx.draw_networkx_nodes(G, pos, nodelist=quiet_nodes, node_size=28, node_color="#CBD5E1", edgecolors="white", linewidths=0.5, ax=ax_g)
    hf = np.array([df_idx.loc[n, "congestion_freq"] * 100 for n in hot_nodes])
    sc = nx.draw_networkx_nodes(G, pos, nodelist=hot_nodes, node_size=55 + 10.5 * hf, node_color=hf, cmap="OrRd", vmin=1, vmax=32, edgecolors="#1E293B", linewidths=0.8, ax=ax_g)
    cbar = plt.colorbar(sc, ax=ax_g, fraction=0.025, pad=0.015); cbar.outline.set_visible(False); cbar.set_label("Tần suất tắc thực (%) — 47 trạm ≥ 1%")

    for n, dx, dy in [(22, -0.22, 0.44), (14, -0.32, -0.26), (4, 0.34, 0.02), (9, 0.34, 0.14), (13, -0.34, -0.18), (65, 0.32, 0.02)]:
        ax_g.annotate(f"N{n} ({df_idx.loc[n, 'congestion_freq']*100:.1f}%)", xy=pos[n], xytext=(pos[n][0] + dx, pos[n][1] + dy), fontsize=8.4, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PAL["coral"], lw=0.9), arrowprops=dict(arrowstyle="->", color=PAL["coral"], lw=1.0))
    ax_g.set_title("(A) Bản đồ mạng lưới có hướng PEMS08 (170 trạm, 277 cạnh — hành lang đậm nối các cụm điểm tắc)", fontweight="bold", pad=8)
    ax_g.axis("off")

    ax_b.grid(True, alpha=0.65); ax_b.set_axisbelow(True)
    x_b, y_c = st_df["betweenness"].to_numpy(), st_df["congestion_freq"].to_numpy() * 100.0
    ax_b.scatter(x_b, y_c, s=40, color="#94A3B8", alpha=0.68, edgecolors="white", lw=0.5, label="Các trạm còn lại (165 nút)")
    ax_b.plot(np.linspace(0, x_b.max(), 40), np.polyval(np.polyfit(x_b, y_c, 1), np.linspace(0, x_b.max(), 40)), color=PAL["slate"], ls="--", lw=1.2, label="Xu hướng tham chiếu (gần phẳng)")
    for i_ex, (nid_ex, dx, dy) in enumerate([(22, 0.006, -0.5), (14, 0.006, 1.5), (13, 0.006, -1.9), (9, 0.006, 0.6), (65, 0.006, -0.8)]):
        r_ex = st_df.loc[st_df["node_id"] == nid_ex].iloc[0]
        bx, cy = float(r_ex["betweenness"]), float(r_ex["congestion_freq"]) * 100.0
        ax_b.scatter([bx], [cy], s=80, color=PAL["coral"], edgecolors="white", lw=1.0, zorder=5, label="Phản ví dụ: tắc nặng dù Betweenness thấp" if i_ex == 0 else None)
        ax_b.annotate(f"N{nid_ex} ({cy:.1f}%)", xy=(bx, cy), xytext=(bx + dx, cy + dy), fontsize=8.2, fontweight="bold", color=PAL["coral"], arrowprops=dict(arrowstyle="-", color=PAL["coral"], lw=0.9))
    ax_b.text(0.97, 0.42, f"Đồ thị TĨNH (N = 170)\nSpearman ρ = {rho_b:.3f}, p = {p_b:.3f}\n→ Chấp nhận H0 (không tương quan)", transform=ax_b.transAxes, ha="right", va="center", fontsize=8.3, bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec=PAL["grid"]))
    ax_b.set_xlabel("Weighted Betweenness Centrality (Tĩnh)"); ax_b.set_ylabel("Tần suất Tắc thực (%)")
    ax_b.set_title("(B) Đồ thị TĨNH: Betweenness không nhận biết được điểm tắc", fontweight="bold")
    ax_b.set_xlim(-0.004, 0.110); ax_b.set_ylim(-1.2, 34.5); ax_b.legend(frameon=False, loc="upper left", fontsize=7.9)

    ax_d.grid(axis="y", alpha=0.65); ax_d.set_axisbelow(True)
    xg, wg = np.arange(len(graph_lrt_df)), 0.20
    v_loc = graph_lrt_df["Cục bộ (14 đặc trưng)"].to_numpy()
    v_in = graph_lrt_df["+ Lân cận In (Đầu vào)"].to_numpy()
    v_out = graph_lrt_df["+ Lân cận Out (Đầu ra)"].to_numpy()
    v_both = graph_lrt_df["+ Cả 2 chiều (In + Out)"].to_numpy()

    ax_d.bar(xg - 1.5 * wg, v_loc, width=wg, color="#CBD5E1", edgecolor="#94A3B8", lw=0.6, label="Chỉ cục bộ (14 đặc trưng)")
    ax_d.bar(xg - 0.5 * wg, v_in, width=wg, color=PAL["amber"], label="+ Lân cận In (Đầu vào)")
    ax_d.bar(xg + 0.5 * wg, v_out, width=wg, color=PAL["teal"], label="+ Lân cận Out (Đầu ra)")
    ax_d.bar(xg + 1.5 * wg, v_both, width=wg, color=PAL["navy"], label="+ Cả 2 chiều (In + Out)")
    for k in range(len(xg)):
        ax_d.text(xg[k] - 1.5 * wg, v_loc[k] + 0.005, f"{v_loc[k]:.3f}", ha="center", va="bottom", fontsize=7.5, color=PAL["slate"])
        ax_d.text(xg[k] + 1.5 * wg, v_both[k] + 0.005, f"{v_both[k]:.3f}\n({graph_lrt_df['Tăng PR-AUC (%)'].iloc[k]})", ha="center", va="bottom", fontsize=7.7, fontweight="bold", color=PAL["navy"])
    ax_d.set_xticks(xg, [f"Δt = {l}\n(p = {p})" for l, p in zip(graph_lrt_df["Tầm cảnh báo (Δt)"], graph_lrt_df["p (LRT)"])], fontsize=8.3)
    ax_d.set_ylim(0, 0.335); ax_d.set_ylabel("Validation PR-AUC (chia 60/20/20)")
    ax_d.set_title("(C) Đồ thị ĐỘNG: Lân cận In & Out nâng mạnh dự báo tắc", fontweight="bold")
    ax_d.legend(frameon=False, loc="upper right", ncol=1, fontsize=8.0)

    _save_and_show(fig, save_path)
