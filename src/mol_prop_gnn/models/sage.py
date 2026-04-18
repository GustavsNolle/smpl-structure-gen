"""GraphSAGE model for molecular property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class MolGraphSAGE(nn.Module):
    """GraphSAGE encoder + graph-level readout for molecular property prediction.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features (unused by GraphSAGE, kept for API compat).
    hidden_dim : int
        Hidden dimension for SAGE layers.
    num_layers : int
        Number of message-passing layers.
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
        hidden_dim: int = 128,
        num_layers: int = 3,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.convs.append(SAGEConv(node_input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

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
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode atom features through SAGE layers."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @property
    def out_channels(self) -> int:
        return self.convs[-1].out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr)
        h = global_mean_pool(h, batch)
        return self.graph_readout(h)
