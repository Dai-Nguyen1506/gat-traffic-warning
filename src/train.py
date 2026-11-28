"""PyTorch L1 (MAE) training loop, vectorized batching, and evaluation for ST-GAT (LSTM -> GATv2)."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.bottlenecks import CongestionThresholds
from src.data import apply_thresholds, inverse_scale_targets
from src.models import STGATv2, count_parameters


def set_seed(seed: int = 42) -> None:
    """Fix all Python, NumPy, and PyTorch seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_vectorized_batches(
    feats_t: torch.Tensor,
    targets_t: torch.Tensor,
    anchors: np.ndarray,
    history_steps: int,
    horizon_steps: int,
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fast vectorized batch generator directly slicing (B, H, N, F) and (B, N, P, 3)
    without per-sample Python DataLoader overhead.
    """
    order = anchors.copy()
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(42)
        rng.shuffle(order)

    h_offs = np.arange(-history_steps + 1, 1, dtype=np.int64)  # (H,)
    p_offs = np.arange(1, horizon_steps + 1, dtype=np.int64)   # (P,)

    for i in range(0, len(order), batch_size):
        b_anc = order[i : i + batch_size]
        x_idx = torch.from_numpy(b_anc[:, None] + h_offs[None, :])  # (B, H)
        y_idx = torch.from_numpy(b_anc[:, None] + p_offs[None, :])  # (B, P)
        xb = feats_t[x_idx]                    # (B, H, N, F)
        yb = targets_t[y_idx].permute(0, 2, 1, 3)  # (B, N, P, 3)
        yield xb, yb


@torch.no_grad()
def predict_anchors(
    model: nn.Module,
    feats_t: torch.Tensor,
    targets_t: torch.Tensor,
    anchors: np.ndarray,
    history_steps: int,
    horizon_steps: int,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference and return standardized predictions & ground truths (M, N, P, 3)."""
    model.eval()
    preds, trues = [], []
    for xb, yb in iter_vectorized_batches(
        feats_t, targets_t, anchors, history_steps, horizon_steps, batch_size, shuffle=False
    ):
        out = model(xb.to(device)).cpu().numpy()
        preds.append(out)
        trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def evaluate_multistep_forecast(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    train_thresholds: CongestionThresholds,
    bottleneck_mask: np.ndarray,
    horizons_steps: Sequence[int] = (3, 6, 12),
) -> Dict[str, object]:
    """
    Evaluate multi-step regression forecast (primary: MAE, companion: RMSE) plus a lightweight
    secondary congestion-marking check using STRICTLY Train-calibrated thresholds.
    Shapes: (M, N, P, 3) with channels [0: Flow, 1: Occupancy, 2: Speed].
    """
    y_true = inverse_scale_targets(y_true_std, target_mean, target_std, clip_physical=False)
    y_pred = inverse_scale_targets(y_pred_std, target_mean, target_std, clip_physical=True)

    true_cong = apply_thresholds(y_true[..., 2], y_true[..., 1], train_thresholds)  # (M, N, P)
    pred_cong = apply_thresholds(y_pred[..., 2], y_pred[..., 1], train_thresholds)

    def _metrics_for_slice(
        yt: np.ndarray,
        yp: np.ndarray,
        yt_s: np.ndarray,
        yp_s: np.ndarray,
        tc: np.ndarray,
        pc: np.ndarray,
    ) -> Dict[str, float]:
        err = yp - yt
        abs_err = np.abs(err)
        bn_abs = abs_err[:, bottleneck_mask]
        return {
            "std_mae": float(np.abs(yp_s - yt_s).mean()),
            "mae_speed": float(abs_err[..., 2].mean()),
            "rmse_speed": float(np.sqrt((err[..., 2] ** 2).mean())),
            "mae_occ": float(abs_err[..., 1].mean()),
            "rmse_occ": float(np.sqrt((err[..., 1] ** 2).mean())),
            "mae_flow": float(abs_err[..., 0].mean()),
            "rmse_flow": float(np.sqrt((err[..., 0] ** 2).mean())),
            "mae_speed_bn": float(bn_abs[..., 2].mean()),
            "mae_occ_bn": float(bn_abs[..., 1].mean()),
            "mae_flow_bn": float(bn_abs[..., 0].mean()),
            "mae_speed_cong": float(abs_err[..., 2][tc].mean()) if tc.any() else float("nan"),
            "mae_occ_cong": float(abs_err[..., 1][tc].mean()) if tc.any() else float("nan"),
            "mae_flow_cong": float(abs_err[..., 0][tc].mean()) if tc.any() else float("nan"),
            "cong_f1": float(f1_score(tc.ravel().astype(int), pc.ravel().astype(int), zero_division=0)),
            "cong_prec": float(precision_score(tc.ravel().astype(int), pc.ravel().astype(int), zero_division=0)),
            "cong_rec": float(recall_score(tc.ravel().astype(int), pc.ravel().astype(int), zero_division=0)),
        }

    overall = _metrics_for_slice(y_true, y_pred, y_true_std, y_pred_std, true_cong, pred_cong)
    by_horizon: Dict[int, Dict[str, float]] = {}
    for k in horizons_steps:
        idx = k - 1
        by_horizon[k * 5] = _metrics_for_slice(
            y_true[:, :, idx : idx + 1],
            y_pred[:, :, idx : idx + 1],
            y_true_std[:, :, idx : idx + 1],
            y_pred_std[:, :, idx : idx + 1],
            true_cong[:, :, idx : idx + 1],
            pred_cong[:, :, idx : idx + 1],
        )

    return {
        "overall": overall,
        "by_horizon": by_horizon,
        "y_true_raw": y_true,
        "y_pred_raw": y_pred,
    }


def evaluate_masked_flow_benchmark(
    y_pred_raw: np.ndarray,
    raw_uncleaned_data: np.ndarray,
    anchors: np.ndarray,
    horizon_steps: int = 12,
    null_val: float = 0.0,
    horizons_steps: Sequence[int] = (3, 6, 12),
) -> Dict[str, object]:
    """
    Compute exact BasicTS / LibCity masked_mae & masked_rmse (masking null_val == 0.0)
    on Flow (channel 0) against the original raw PeMS ground truth for fair literature comparison.
    y_pred_raw has shape (M, N, P, 3).
    """
    p_offs = np.arange(1, horizon_steps + 1, dtype=np.int64)
    y_idx = anchors[:, None] + p_offs[None, :]  # (M, P)
    # raw_flow: (M, N, P)
    raw_flow = np.transpose(raw_uncleaned_data[y_idx, :, 0], (0, 2, 1)).astype(np.float32)
    pred_flow = y_pred_raw[..., 0].astype(np.float32)

    def _m_stats(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
        mask = yt > null_val
        diff = yp[mask] - yt[mask]
        return {
            "masked_mae_flow": float(np.abs(diff).mean()),
            "masked_rmse_flow": float(np.sqrt((diff ** 2).mean())),
            "valid_ratio_pct": float(mask.mean() * 100.0),
        }

    overall = _m_stats(raw_flow, pred_flow)
    by_horizon: Dict[int, Dict[str, float]] = {}
    for k in horizons_steps:
        idx = k - 1
        by_horizon[k * 5] = _m_stats(raw_flow[:, :, idx : idx + 1], pred_flow[:, :, idx : idx + 1])

    return {"overall": overall, "by_horizon": by_horizon}


def train_stgatv2_experiment(
    bundle: Dict[str, object],
    config_name: str,
    feature_indices: Sequence[int],
    hidden_dim: int = 32,
    num_lstm_layers: int = 1,
    num_gat_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.10,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 4e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
    load_if_exists: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train (or load) a single ST-GAT (LSTM -> GATv2) configuration with pure L1 (MAE) loss."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = int(bundle["history_steps"])
    P = int(bundle["horizon_steps"])
    feats_t = torch.from_numpy(
        np.ascontiguousarray(bundle["feats_scaled"][:, :, list(feature_indices)], dtype=np.float32)
    )
    targets_t = torch.from_numpy(np.ascontiguousarray(bundle["targets_scaled"], dtype=np.float32))

    model = STGATv2(
        in_channels=len(feature_indices),
        edge_index=bundle["edge_index"],
        edge_type=bundle["edge_type"],
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        horizon_steps=P,
        out_vars=3,
        dropout=dropout,
    ).to(device)

    sp = Path(save_path) if save_path is not None else None
    if load_if_exists and sp is not None and sp.exists():
        ckpt = torch.load(sp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        history = ckpt.get("history", {"train_loss": [], "val_loss": []})
        best_val_loss = float(ckpt["best_val_loss"])
        best_epoch = int(ckpt["best_epoch"])
        elapsed = float(ckpt.get("train_seconds", 0.0))
    else:
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=2e-4
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        wait = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            tr_sum, tr_n = 0.0, 0
            for xb, yb in iter_vectorized_batches(
                feats_t, targets_t, bundle["t_train"], H, P, batch_size, shuffle=True, rng=rng
            ):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                bs = xb.shape[0]
                tr_sum += float(loss.item()) * bs
                tr_n += bs
            train_loss = tr_sum / max(tr_n, 1)

            model.eval()
            va_sum, va_n = 0.0, 0
            with torch.no_grad():
                for xb, yb in iter_vectorized_batches(
                    feats_t, targets_t, bundle["t_val"], H, P, batch_size * 2, shuffle=False
                ):
                    xb, yb = xb.to(device), yb.to(device)
                    loss = criterion(model(xb), yb)
                    bs = xb.shape[0]
                    va_sum += float(loss.item()) * bs
                    va_n += bs
            val_loss = va_sum / max(va_n, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        elapsed = time.perf_counter() - t0
        model.load_state_dict(best_state)

        if sp is not None:
            sp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config_name": config_name,
                    "feature_indices": list(feature_indices),
                    "hidden_dim": hidden_dim,
                    "num_lstm_layers": num_lstm_layers,
                    "num_gat_layers": num_gat_layers,
                    "num_heads": num_heads,
                    "horizon_steps": P,
                    "dropout": dropout,
                    "state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "history": history,
                    "train_seconds": elapsed,
                },
                sp,
            )

    y_pred_val_std, y_true_val_std = predict_anchors(
        model, feats_t, targets_t, bundle["t_val"], H, P, device, batch_size=256
    )
    y_pred_test_std, y_true_test_std = predict_anchors(
        model, feats_t, targets_t, bundle["t_test"], H, P, device, batch_size=256
    )

    val_eval = evaluate_multistep_forecast(
        y_true_val_std,
        y_pred_val_std,
        bundle["target_mean"],
        bundle["target_std"],
        bundle["train_thresholds"],
        bundle["bottleneck_mask"],
    )
    test_eval = evaluate_multistep_forecast(
        y_true_test_std,
        y_pred_test_std,
        bundle["target_mean"],
        bundle["target_std"],
        bundle["train_thresholds"],
        bundle["bottleneck_mask"],
    )

    if verbose:
        ov = test_eval["overall"]
        print(
            f"[{config_name:18s}] params={count_parameters(model):,} | "
            f"ep={len(history['val_loss']):2d} (best={best_epoch:2d}) | "
            f"val_L1={best_val_loss:.4f} | "
            f"Test Avg MAE: Spd={ov['mae_speed']:.3f}, Occ={ov['mae_occ']:.5f}, Flw={ov['mae_flow']:.2f}, "
            f"Std={ov['std_mae']:.4f} | CongSpd={ov['mae_speed_cong']:.3f} | {elapsed:.1f}s"
        )

    return {
        "config_name": config_name,
        "feature_indices": list(feature_indices),
        "n_params": count_parameters(model),
        "epochs_run": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_seconds": elapsed,
        "history": history,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "model": model,
    }
