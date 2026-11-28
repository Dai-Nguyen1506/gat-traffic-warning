"""Pure-PyTorch inductive ST-GAT (2-layer LSTM -> Multi-Head GATv2) for multi-step PeMS forecasting.

Inherits the ST-GAT framework (Zhang, Yu & Liu, IEEE Access 2019), compressing the 12-step
historical sequence per node via a shared-weight 2-layer LSTM before applying dynamic
directed Multi-Head GATv2 attention (Brody et al., ICLR 2022) across Self / In / Out edges.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Vectorized Multi-Head GATv2 layer (Brody et al., ICLR 2022) with typed directed edges."""

    def __init__(
        self,
        in_dim: int,
        head_dim: int = 16,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.10,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.out_dim = head_dim * num_heads
        self.negative_slope = negative_slope

        self.w_src = nn.Linear(in_dim, self.out_dim, bias=False)
        self.w_dst = nn.Linear(in_dim, self.out_dim, bias=False)
        self.edge_bias = nn.Embedding(num_edge_types, self.out_dim)
        nn.init.zeros_(self.edge_bias.weight)

        self.attn_vec = nn.Parameter(torch.empty(1, 1, num_heads, head_dim))
        nn.init.xavier_uniform_(self.attn_vec)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, in_dim)
            edge_index: (2, E) int64 [src, dst]
            edge_type: (E,) int64 in {0: Self, 1: In, 2: Out}
        Returns:
            out: (B, N, num_heads * head_dim)
            alpha_mean: (B, E) head-averaged dynamic attention weights
        """
        B, N, _ = h.shape
        K, D = self.num_heads, self.head_dim
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_src = self.w_src(h).view(B, N, K, D)  # (B, N, K, D)
        h_dst = self.w_dst(h).view(B, N, K, D)  # (B, N, K, D)
        e_bias = self.edge_bias(edge_type).view(1, E, K, D)

        # Dynamic GATv2 attention: apply LeakyReLU BEFORE inner product with attn_vec
        z = F.leaky_relu(h_src[:, src] + h_dst[:, dst] + e_bias, negative_slope=self.negative_slope)
        scores = (z * self.attn_vec).sum(dim=-1)  # (B, E, K)

        # Numerically stable segment softmax grouped by destination node dst
        dst_exp = dst.view(1, E, 1).expand(B, E, K)
        max_per_dst = torch.full((B, N, K), -1e9, dtype=scores.dtype, device=scores.device)
        max_per_dst.scatter_reduce_(1, dst_exp, scores.detach(), reduce="amax", include_self=False)
        exp_scores = torch.exp(scores - max_per_dst.gather(1, dst_exp))

        denom = torch.zeros((B, N, K), dtype=scores.dtype, device=scores.device)
        denom.scatter_add_(1, dst_exp, exp_scores)
        alpha = exp_scores / (denom.gather(1, dst_exp) + 1e-12)  # (B, E, K)

        alpha_w = self.attn_drop(alpha)
        msg = h_src[:, src] * alpha_w.unsqueeze(-1)  # (B, E, K, D)

        out = torch.zeros((B, N, K, D), dtype=h.dtype, device=h.device)
        dst_msg = dst.view(1, E, 1, 1).expand(B, E, K, D)
        out.scatter_add_(1, dst_msg, msg)
        out = out.reshape(B, N, K * D)

        alpha_mean = alpha.mean(dim=-1)  # (B, E)
        return out, alpha_mean


class STGATv2(nn.Module):
    """
    Inductive Spatial-Temporal Graph Attention Network (2-layer LSTM -> Multi-Head GATv2 -> Residual FC).
    Shared across all nodes so weights transfer directly from PEMS08 (N=170) to PEMS04 (N=307).
    """

    def __init__(
        self,
        in_channels: int,
        edge_index: np.ndarray,
        edge_type: np.ndarray,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        horizon_steps: int = 12,
        out_vars: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.horizon_steps = horizon_steps
        self.out_vars = out_vars

        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("edge_type", torch.as_tensor(edge_type, dtype=torch.long))

        # 1. Temporal compression: shared-weight LSTM over H=12 past steps (batch_first=False avoids CPU transpose copy)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=False,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.temp_norm = nn.LayerNorm(hidden_dim)

        # 2. Spatial dynamic attention: stacked Multi-Head GATv2 over directed graph (Self / In / Out)
        head_dim = hidden_dim // num_heads
        self.gat_layers = nn.ModuleList(
            [
                GATv2Layer(
                    in_dim=hidden_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_edge_types=3,
                    dropout=dropout,
                )
                for _ in range(num_gat_layers)
            ]
        )
        self.gat_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])
        self.drop = nn.Dropout(dropout)

        # 3. Direct multi-step regression readout (residual from last observed [Flow, Occ, Speed])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * out_vars),
        )

    def set_graph(self, edge_index: np.ndarray, edge_type: np.ndarray) -> None:
        """Swap graph topology (e.g., PEMS08 -> PEMS04) while preserving learned weights."""
        device = self.edge_index.device
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        self.edge_type = torch.as_tensor(edge_type, dtype=torch.long, device=device)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H, N, C_in) with channels 0..2 always standardized [Flow, Occ, Speed].
        Returns:
            y_pred: (B, N, horizon_steps, 3) standardized predictions
            alpha_last: (B, E) dynamic GATv2 attention weights from the final spatial layer
        """
        B, H, N, C = x.shape
        last_raw = x[:, -1, :, : self.out_vars]  # (B, N, 3)

        # Compress temporal history per node via LSTM: (H, B * N, C)
        x_seq = x.permute(1, 0, 2, 3).reshape(H, B * N, C).contiguous()
        _, (h_n, _) = self.lstm(x_seq)
        h = self.temp_norm(h_n[-1].reshape(B, N, self.hidden_dim))

        # Dynamic spatial propagation via Multi-Head GATv2
        alpha_last = None
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_sp, alpha_last = gat(h, self.edge_index, self.edge_type)
            h = norm(h + self.drop(F.elu(h_sp)))

        # Residual multi-step forecast
        delta = self.head(h).view(B, N, self.horizon_steps, self.out_vars)
        y_pred = last_raw.unsqueeze(2) + delta
        return y_pred, alpha_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred, _ = self.forward_with_attention(x)
        return y_pred


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
