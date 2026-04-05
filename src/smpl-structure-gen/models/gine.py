"""GINEConv-based Graph Neural Network for Resilient Heterogeneous Message Passing.

Designed specifically to prevent 'Dead ReLU' feature collapse during contagion shocks
by robustly projecting structural node and edge attributes continuously.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class TradeFlowGINE(nn.Module):
    """GINE network that perfectly mirrors the biological structural flow of Contagion.
    
    Parameters
    ----------
    node_input_dim : int
        Dimension of the macroscopic node features.
    edge_input_dim : int
        Dimension of the edge trade-flow features (Continuous volume indicators).
    hidden_dim : int
        Size of message passing hidden state.
    num_gnn_layers : int
        Total number of GINE hops.
    decoder_hidden_dim : int
        Readout layer width.
    dropout : float
        Standard dropout fraction.
    output_dim : int
        Continuous vector dimension mapping node vulnerability targets.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # Initialize LeakyReLU GINE layers
        for _ in range(num_gnn_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_dim, hidden_dim)
                    ), 
                    edge_dim=hidden_dim
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = dropout

        # 1. The pure Tabular MLP Baseline (Perfectly mirrors `MLPBaseline`)
        self.mlp_baseline = nn.Sequential(
            nn.Linear(node_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

        # 2. The Topological Contagion Readout
        self.topo_readout = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim)
        )
        
        # 3. ReZero Scale Parameter
        # Completely blocks noisy graphs from destroying early epoch tabular gradients.
        self.topo_scale = nn.Parameter(torch.zeros(1))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode through multi-hop topological contagion structures."""
        h = F.leaky_relu(self.node_proj(x), 0.2)
        e = F.leaky_relu(self.edge_proj(edge_attr), 0.2)
        
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, edge_index, e)
            h_next = norm(h_next)
            h_next = F.leaky_relu(h_next, 0.2)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            h = h + h_next  # Residual connection prevents message wash-out
            
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Calculate Node Contagion Reaction Deltas."""
        # Baseline Tabular Prediction
        y_tabular = self.mlp_baseline(x)
        
        # Structural Topological Ripple
        h_graph = self.encode(x, edge_index, edge_attr)
        y_topo = self.topo_readout(h_graph)
        
        # Superposition specifically modulated by the ReZero Learnable Gate
        return y_tabular + (self.topo_scale * y_topo)
