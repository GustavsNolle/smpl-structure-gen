"""Constraint-based joint embedder for multi-dataset molecular property prediction.

This architecture acts as a 'Multi-Dimensional Map'. 
It forces the molecular representation through a small bottleneck equal to the
number of target tasks (datasets), enforcing that each dimension corresponds
to a unique chemical property (e.g. Dim 1 = BBBP, Dim 2 = ESOL).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class JointMolEmbedder(nn.Module):
    """GNN encoder with a constrained semantic bottleneck mapping.

    Parameters
    ----------
    backbone : nn.Module
        The underlying GNN encoder (GCN, GAT, GINE, etc.).
        Must implement an `encode(x, edge_index, edge_attr, batch, ...)` method.
    backbone_out_dim : int
        Dimension of the graph-level embedding produced by the backbone.
    num_datasets : int
        Size of the semantic bottleneck (must equal the number of datasets).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        num_datasets: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_datasets = num_datasets
        self.dropout = dropout

        # The Semantic Bottleneck Map Layer
        # Compresses the high-dim graph embedding strictly down into N dimensions
        # where N = number of datasets (chemical properties)
        self.semantic_map_layer = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim // 2, num_datasets)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.
        
        Returns
        -------
        h_semantic : (B, num_datasets)
            The unactivated predictions serving as our semantic map footprint.
            Each dimension corresponds directly to a chemical property.
        """
        # Encode graph structure via backbone
        # We assume the backbone has an encode() method that returns node embeddings
        h_node = self.backbone.encode(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        
        # Pool to graph level (B, backbone_out_dim)
        h_graph = global_mean_pool(h_node, batch)
        
        # Project through semantic bottleneck (B, num_datasets)
        h_semantic = self.semantic_map_layer(h_graph)
        
        return h_semantic
