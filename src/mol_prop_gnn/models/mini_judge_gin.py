"""Lightweight GIN with Residual Connections and Virtual Node.

A compact GIN variant designed for fast, accurate single-property prediction.
The virtual node aggregates global graph information at each layer, dramatically
improving long-range communication in molecular graphs.

Reference: Gilmer et al. (2020) "Strategies for Pre-training Graph Neural Networks"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class ResGINBlock(nn.Module):
    """GIN block with residual connection and pre-norm."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(nn=self.mlp, train_eps=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.conv(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return x + h  # Residual connection


class MiniJudgeGIN(nn.Module):
    """Compact GIN with Virtual Node for single-property molecular regression.
    
    The virtual node acts as a global scratchpad: at each GIN layer, every
    real node reads from and writes to the virtual node, enabling O(1) 
    information flow across the entire graph regardless of diameter.
    
    Architecture:
      Input → Linear Projection → [ResGIN + VirtualNode] × N → Pool → MLP → Scalar
    """

    def __init__(
        self,
        node_input_dim: int = 38,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # GIN blocks with residual connections
        self.blocks = nn.ModuleList([
            ResGINBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # Virtual Node MLPs: project aggregated real-node info into VN, and vice versa
        self.vn_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.BatchNorm1d(hidden_dim)

        # Prediction head
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        h = self.node_proj(x)

        # Initialize virtual node embedding as zeros (one per graph)
        vn_emb = torch.zeros(num_graphs, self.hidden_dim, device=x.device)

        for layer_idx, block in enumerate(self.blocks):
            # 1. Broadcast virtual node → real nodes (additive injection)
            h = h + vn_emb[batch]

            # 2. GIN message passing with residual
            h = block(h, edge_index)

            # 3. Aggregate real nodes → virtual node (with residual on VN)
            h_pool = global_mean_pool(h, batch)  # [num_graphs, hidden_dim]
            vn_emb = vn_emb + self.vn_encoder[layer_idx](h_pool)

        # Final readout
        h = self.final_norm(h)
        h_graph = global_mean_pool(h, batch)

        return self.readout(h_graph).squeeze(-1)
