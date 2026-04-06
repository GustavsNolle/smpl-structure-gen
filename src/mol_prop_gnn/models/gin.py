"""GIN (Graph Isomorphism Network) for molecular property prediction.

GIN is mathematically proven to be as powerful as the 1-WL test for graph isomorphism,
making it significantly more expressive for structural patterns than GCN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINBlock(nn.Module):
    """A GIN message passing block with a 2-layer MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        train_eps: bool = True,
    ) -> None:
        super().__init__()
        # Xu et al. (2018) recommends a 2-layer MLP for the GIN kernel
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(nn=self.mlp, train_eps=train_eps)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MolGIN(nn.Module):
    """GIN encoder for molecular graphs.
    
    Note: This pure GIN implementation does not use edge features (bonds) 
    directly in the convolutions, focusing on node-level isomorphism.
    """

    def __init__(
        self,
        node_input_dim: int = 38,
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
                GINBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
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
        """Encode atom features through GIN layers."""
        h = self.node_proj(x)
        for block in self.blocks:
            h = block(h, edge_index)
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
