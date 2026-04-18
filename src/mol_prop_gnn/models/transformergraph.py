"""Transformer + GNN hybrid for molecular property prediction.
Replaces global mean pooling with cross-attention between nodes,
preserving pairwise structural information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class CrossAttentionPool(nn.Module):
    """Cross-attention pooling: attend from learnable query to all nodes."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # x: (N, D), batch: (N,)
        # Pad to max nodes per graph
        max_nodes = torch.bincount(batch).max().item()
        padded = []
        masks = []

        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            graph_x = x[mask]
            pad_len = max_nodes - graph_x.size(0)
            if pad_len > 0:
                graph_x = torch.cat([graph_x, torch.zeros(pad_len, x.size(1), device=x.device)])
                mask_pad = torch.cat([torch.ones(graph_x.size(0) - pad_len, device=x.device),
                                      torch.zeros(pad_len, device=x.device)])
            else:
                mask_pad = torch.ones(graph_x.size(0), device=x.device)
            padded.append(graph_x.unsqueeze(0))
            masks.append(mask_pad.unsqueeze(0))

        padded = torch.cat(padded, dim=0)  # (B, max_nodes, D)
        masks = torch.cat(masks, dim=0).bool()  # (B, max_nodes)

        # Expand query to batch
        query = self.query.expand(padded.size(0), -1, -1)  # (B, 1, D)

        # Cross-attention: query attends to nodes
        attended, _ = self.attention(query, padded, padded, key_padding_mask=~masks)
        return attended.squeeze(1)  # (B, D)


class EdgeAwareBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, edge_input_dim: int, dropout: float):
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_input_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h + self.residual(x))


class MolTransformerGNN(nn.Module):
    """GNN backbone + Cross-Attention Pooling + MLP readout."""

    def __init__(
        self,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 8,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(EdgeAwareBlock(node_input_dim, hidden_dim, edge_input_dim, dropout))
        for _ in range(num_gnn_layers - 1):
            self.blocks.append(EdgeAwareBlock(hidden_dim, hidden_dim, edge_input_dim, dropout))

        self.pool = CrossAttentionPool(hidden_dim, num_attention_heads)
        self.dropout = dropout

        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def encode(self, x, edge_index, edge_attr):
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        h = self.encode(x, edge_index, edge_attr)
        h = self.pool(h, batch)
        return self.graph_readout(h)
