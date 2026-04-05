"""RGCN-based model for molecular property prediction.

Uses relation-typed message passing where relations correspond to
bond types (single, double, triple, aromatic).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool


class MolRGCN(nn.Module):
    """Relational GCN encoder + graph-level readout.

    Relations correspond to bond types:
    0=single, 1=double, 2=triple, 3=aromatic.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features (unused by RGCN, kept for API compat).
    hidden_dim : int
        Hidden dimension for RGCN layers.
    num_layers : int
        Number of RGCN message-passing layers.
    num_relations : int
        Number of bond type relations.
    decoder_hidden_dim : int
        Hidden dimension in readout MLP.
    dropout : float
        Dropout probability.
    output_dim : int
        Number of output tasks.
    """

    def __init__(
        self,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_relations: int = 4,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        self.convs.append(RGCNConv(node_input_dim, hidden_dim, num_relations))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.residuals.append(
            nn.Identity() if node_input_dim == hidden_dim
            else nn.Linear(node_input_dim, hidden_dim)
        )

        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.residuals.append(nn.Identity())

        self.dropout = dropout

        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def encode(self, x, edge_index, edge_attr=None, edge_type=None, **kwargs):
        if edge_type is None and edge_attr is not None:
            # Fallback for RGCN which typically uses typed edges
            edge_type = torch.zeros(edge_attr.size(0), dtype=torch.long, device=edge_attr.device)

        for conv, norm, residual in zip(self.convs, self.norms, self.residuals):
            h = conv(x, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(h + residual(x))
    @property
    def out_channels(self) -> int:
        """Dimension of node embeddings after encoding."""
        return self.norms[-1].normalized_shape[0] if self.norms else self.node_input_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None, edge_type=None, **kwargs):
        h = self.encode(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, **kwargs)
        h = global_mean_pool(h, batch)
        return self.graph_readout(h)
