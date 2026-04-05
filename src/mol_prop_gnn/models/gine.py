"""GINEConv-based Graph Neural Network for molecular property prediction.

Uses a ReZero-gated superposition of tabular (fingerprint-like) and
topological (message-passing) pathways for robust graph-level prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class MolGINE(nn.Module):
    """GINE network with ReZero-gated tabular+topological superposition.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features.
    hidden_dim : int
        Size of message passing hidden state.
    num_gnn_layers : int
        Total number of GINE hops.
    decoder_hidden_dim : int
        Readout layer width.
    dropout : float
        Standard dropout fraction.
    output_dim : int
        Number of prediction tasks.
    """

    def __init__(
        self,
        node_input_dim: int = 39,
        edge_input_dim: int = 13,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=hidden_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

        # Tabular pathway: operates on atom features directly (no message passing)
        self.mlp_baseline = nn.Sequential(
            nn.Linear(node_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

        # Topological pathway: reads out from GNN atom embeddings
        self.topo_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

        # ReZero gate: starts at zero, learns to blend topological signal
        self.topo_scale = nn.Parameter(torch.zeros(1))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom features through multi-hop message passing."""
        h = F.leaky_relu(self.node_proj(x), 0.2)
        e = F.leaky_relu(self.edge_proj(edge_attr), 0.2)

        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, edge_index, e)
            h_next = norm(h_next)
            h_next = F.leaky_relu(h_next, 0.2)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            h = h + h_next  # Residual connection

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with ReZero-gated superposition.

        Combines a tabular baseline (mean-pooled atom features → MLP)
        with a topological signal (GNN embeddings → pool → readout).
        """
        # Tabular pathway: pool raw atom features → predict
        x_pooled = global_mean_pool(x, batch)
        y_tabular = self.mlp_baseline(x_pooled)

        # Topological pathway: GNN encode → pool → predict
        h_graph = self.encode(x, edge_index, edge_attr)
        h_pooled = global_mean_pool(h_graph, batch)
        y_topo = self.topo_readout(h_pooled)

        # ReZero superposition
        return y_tabular + (self.topo_scale * y_topo)
