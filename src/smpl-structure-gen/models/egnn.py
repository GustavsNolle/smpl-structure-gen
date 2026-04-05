"""EGNN-based model for edge-level trade flow regression.

This model uses edge features to modulate the messages between nodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """Simple Edge-conditioned Message Passing layer."""

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr='mean')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_channels, out_channels),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_channels)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate [src, dst, edge_feat]
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        return msg * self.gate_mlp(edge_attr)

    def update(self, aggr_out, x):
        # Combine original node feature with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=-1)
        out = self.node_mlp(update_input)
        return self.norm(out + self.residual(x))


class TradeFlowEGNN(nn.Module):
    """EGNN encoder + edge-level MLP decoder."""

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(EGNNLayer(node_input_dim, hidden_dim, edge_input_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(EGNNLayer(hidden_dim, hidden_dim, edge_input_dim))

        self.dropout = dropout

        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim)
        )

    def encode(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr, **kwargs):
        h = self.encode(x, edge_index, edge_attr)
        return self.node_readout(h)
