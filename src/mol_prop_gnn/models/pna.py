"""PNA (Principal Neighbourhood Aggregation) for molecular property prediction.

Uses multiple aggregators (mean, max, min, std) and degree-scaled scalers to 
capture high-order graph statistics, making it more expressive than standard 
GNNs for molecular patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool


class PNABlock(nn.Module):
    """A PNA message passing block with multiple aggregators and scalers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        deg: torch.Tensor,
        edge_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # PNA aggregates: mean, max, min, std
        aggregators = ["mean", "max", "min", "std"]
        # PNA scalers: identity, amplification, attenuation
        scalers = ["identity", "amplification", "attenuation"]

        self.conv = PNAConv(
            in_channels=in_dim,
            out_channels=out_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MolPNA(nn.Module):
    """PNA encoder for molecular graphs.

    Parameters
    ----------
    deg : torch.Tensor
        The degree distribution (histogram) of the training set.
        Used for power-norm scaling in PNAConv.
    """

    def __init__(
        self,
        deg: torch.Tensor,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 256,
        num_gnn_layers: int = 5,
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.blocks.append(
                PNABlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    deg=deg,
                    edge_dim=edge_input_dim,
                    dropout=dropout,
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """Encode atom features through PNA layers."""
        h = self.node_proj(x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        return h

    @property
    def out_channels(self) -> int:
        return self.blocks[-1].norm.num_features if self.blocks else self.node_input_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr)
        h_pooled = global_mean_pool(h, batch)
        return self.readout(h_pooled)
