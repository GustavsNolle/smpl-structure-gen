"""MLP baseline for trade flow prediction (no graph structure).

This model takes the same features as the GCN (source node features,
destination node features, edge features) but does NOT use message
passing. It serves as a control to measure whether graph structure
actually helps beyond simple feature concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Feedforward MLP baseline for edge-level regression.

    Takes ``[x_src ‖ x_dst ‖ edge_attr]`` → scalar prediction.

    Parameters
    ----------
    input_dim : int
        Total input dimension (2 * node_feat_dim + edge_feat_dim).
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: pure tabular execution on Node macro features."""
        return self.mlp(x)
