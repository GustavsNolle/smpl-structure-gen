"""GAT-based model for edge-level trade flow regression.

Architecture
------------
1. Encode node features through stacked edge-aware GATv2 layers (multi-head attention).
2. For each edge (i → j), concatenate [h_i ‖ h_j ‖ edge_attr].
3. Pass through an MLP decoder → scalar predicted log-trade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class EdgeDecoder(nn.Module):
    """MLP that maps rich edge interactions to a scalar prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Predict trade flow for each edge."""
        x = torch.cat(
            [h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, edge_attr],
            dim=-1,
        )
        return self.mlp(x).squeeze(-1)


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


class TradeFlowGAT(nn.Module):
    """GAT encoder + edge-level MLP decoder for trade forecasting.

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node features.
    edge_input_dim : int
        Dimension of edge features.
    hidden_dim : int
        Hidden dimension for GAT layers (per head).
    num_gnn_layers : int
        Number of GAT message-passing layers.
    heads : int
        Number of attention heads.
    decoder_hidden_dim : int
        Hidden dimension in the edge decoder MLP.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        heads: int = 4,
        decoder_hidden_dim: int = 32,
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
        self.node_readout = nn.Sequential(
            nn.Linear(out_gnn_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim)
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features through GAT layers."""
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Full forward pass."""
        h = self.encode(x, edge_index, edge_attr)
        return self.node_readout(h)
