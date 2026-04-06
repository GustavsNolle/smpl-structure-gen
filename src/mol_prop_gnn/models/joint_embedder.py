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
        bottleneck_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_datasets = num_datasets
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        # The High-Capacity Shared Bottleneck
        # Projects backbone embedding to a large 256-dim latent space
        self.semantic_bottleneck = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction heads for each task
        self.task_head = nn.Linear(bottleneck_dim, num_datasets)

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
        h_node = self.backbone.encode(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        
        # Pool to graph level (B, backbone_out_dim)
        h_graph = global_mean_pool(h_node, batch)
        
        # Project through high-capacity bottleneck
        h_shared = self.semantic_bottleneck(h_graph)
        
        # Final prediction (B, num_datasets)
        h_semantic = self.task_head(h_shared)
        
        return h_semantic

    def re_initialize_map_layer(self, num_tasks: int, backbone_out_dim: int) -> None:
        """Replace the prediction head for a new task count (e.g. during finetuning)."""
        self.num_datasets = num_tasks
        # We keep the bottleneck, but swap the final head
        self.task_head = nn.Linear(self.bottleneck_dim, num_tasks)
