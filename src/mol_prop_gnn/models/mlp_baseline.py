"""MLP baseline for molecular property prediction (no graph structure).

Takes mean-pooled atom features and predicts molecular properties
without using any message passing. Serves as a control to measure
whether graph structure actually helps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class MLPBaseline(nn.Module):
    """Feedforward MLP baseline for graph-level prediction.

    Pools atom features via global mean, then predicts through an MLP.

    Parameters
    ----------
    input_dim : int
        Atom feature dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout probability.
    output_dim : int
        Number of output tasks.
    """

    def __init__(
        self,
        input_dim: int = 39,
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
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: pool atom features → MLP prediction."""
        h = global_mean_pool(x, batch)
        return self.mlp(h)
