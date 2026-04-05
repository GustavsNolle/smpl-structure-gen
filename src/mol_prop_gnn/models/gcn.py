"""GCN-based model for molecular property prediction.

Architecture
------------
1. Encode atom features through stacked edge-aware GINE layers → atom embeddings.
2. Apply global mean pooling → graph-level embedding.
3. Pass through an MLP readout → molecular property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class EdgeAwareBlock(nn.Module):
    """Residual edge-aware message passing block."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_input_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_input_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h + self.residual(x))


class MolGCN(nn.Module):
    """Edge-aware GNN encoder + graph-level readout for molecular property prediction.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features.
    hidden_dim : int
        Hidden dimension for GNN layers.
    num_gnn_layers : int
        Number of message-passing layers.
    decoder_hidden_dim : int
        Hidden dimension in the readout MLP.
    dropout : float
        Dropout probability.
    output_dim : int
        Number of output tasks (1 for single-task).
    """

    def __init__(
        self,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            EdgeAwareBlock(
                in_dim=node_input_dim,
                hidden_dim=hidden_dim,
                edge_input_dim=edge_input_dim,
                dropout=dropout,
            )
        )
        for _ in range(num_gnn_layers - 1):
            self.blocks.append(
                EdgeAwareBlock(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_input_dim=edge_input_dim,
                    dropout=dropout,
                )
            )

        self.dropout = dropout

        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
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
        """Encode atom features through GNN layers."""
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x

    @property
    def out_channels(self) -> int:
        """Dimension of node embeddings after encoding."""
        return self.blocks[-1].norm.normalized_shape[0] if self.blocks else self.node_input_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Full forward pass: encode atoms → pool → predict."""
        h = self.encode(x, edge_index, edge_attr)
        h = global_mean_pool(h, batch)
        return self.graph_readout(h)
