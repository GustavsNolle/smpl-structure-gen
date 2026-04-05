"""RGCN-based model for trade flow regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class TradeFlowRGCN(nn.Module):
    """RGCN encoder + edge-level MLP decoder.
    
    In this implementation, relations correspond to BACI HS product categories.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_relations: int = 21,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        # First layer
        self.convs.append(RGCNConv(node_input_dim, hidden_dim, num_relations))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.residuals.append(nn.Identity() if node_input_dim == hidden_dim else nn.Linear(node_input_dim, hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.residuals.append(nn.Identity())

        self.dropout = dropout

        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim)
        )

    def encode(self, x, edge_index, edge_type):
        for conv, norm, residual in zip(self.convs, self.norms, self.residuals):
            h = conv(x, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(h + residual(x))
        return x

    def forward(self, x, edge_index, edge_attr=None, edge_type=None, **kwargs):
        if edge_type is None and edge_attr is not None:
            edge_type = torch.zeros(edge_attr.size(0), dtype=torch.long, device=edge_attr.device)
            
        h = self.encode(x, edge_index, edge_type)
        return self.node_readout(h)
