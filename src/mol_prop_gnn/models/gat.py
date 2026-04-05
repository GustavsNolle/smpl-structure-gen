"""GAT-based model for molecular property prediction.

Architecture
------------
1. Encode atom features through stacked GATv2 layers with multi-head attention.
2. Apply global mean pooling → graph-level embedding.
3. Pass through an MLP readout → molecular property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class EdgeAwareGATBlock(nn.Module):
    """Residual edge-aware GAT block."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        edge_input_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_input_dim,
        )
        out_dim = hidden_dim * heads
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h + self.residual(x))


class MolGAT(nn.Module):
    """GAT encoder + graph-level readout for molecular property prediction.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features.
    hidden_dim : int
        Hidden dimension per attention head.
    num_gnn_layers : int
        Number of GAT message-passing layers.
    heads : int
        Number of attention heads.
    decoder_hidden_dim : int
        Hidden dimension in the readout MLP.
    dropout : float
        Dropout probability.
    output_dim : int
        Number of output tasks.
    """

    def __init__(
        self,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        heads: int = 4,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            EdgeAwareGATBlock(
                in_dim=node_input_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                edge_input_dim=edge_input_dim,
                dropout=dropout,
            )
        )
        out_dim = hidden_dim * heads
        for _ in range(num_gnn_layers - 1):
            self.blocks.append(
                EdgeAwareGATBlock(
                    in_dim=out_dim,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    edge_input_dim=edge_input_dim,
                    dropout=dropout,
                )
            )

        self.dropout = dropout

        out_gnn_dim = hidden_dim * heads
        self.graph_readout = nn.Sequential(
            nn.Linear(out_gnn_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom features through GAT layers."""
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x

    @property
    def out_channels(self) -> int:
        """Dimension of node embeddings after encoding."""
        return self.blocks[-1].conv.out_channels * self.blocks[-1].conv.heads

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Full forward pass: encode → pool → predict."""
        h = self.encode(x, edge_index, edge_attr)
        h = global_mean_pool(h, batch)
        return self.graph_readout(h)
