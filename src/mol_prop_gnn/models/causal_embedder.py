"""Causal Information Bottleneck embedder for multi-dataset molecular property prediction.

This architecture wraps a backbone to predict a continuous node mask, splitting the
graph into a Causal Subgraph (Pharmacophore) and an Environmental Subgraph (Scaffold).
Both subgraphs are then routed through to prediction heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

class CausalMolEmbedder(nn.Module):
    """GNN encoder with a Causal Subgraph Extraction mechanism.

    Parameters
    ----------
    backbone : nn.Module
        The underlying GNN encoder (GCN, PNA, etc.).
        Must implement an `encode(x, edge_index, edge_attr, batch, ...)` method.
    backbone_out_dim : int
        Dimension of the graph-level embedding produced by the backbone.
    num_datasets : int
        Size of the semantic bottleneck (must equal the number of datasets).
    bottleneck_dim : int
        Dimension of the shared projection layer before task heads.
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

        # Extractor network: Predicts probability of a node belonging to Causal subgraph
        self.extractor = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_dim // 2, 1)
            # We don't apply sigmoid here, we apply it dynamically in the forward pass
            # to prevent saturation during initialization.
        )

        # The High-Capacity Shared Bottleneck for Causal Subgraph
        self.causal_bottleneck = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Environmental bottleneck (often shared, but keeping separate prevents spurious shortcuts)
        self.env_bottleneck = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction heads
        self.causal_head = nn.Linear(bottleneck_dim, num_datasets)
        self.env_head = nn.Linear(bottleneck_dim, num_datasets)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns
        -------
        pred_c : (B, num_datasets)
            Predictions based strictly on the Causal subgraph.
        pred_e : (B, num_datasets)
            Predictions based strictly on the Environment subgraph.
        mask : (N, 1)
            The node probability mask used for the split.
        """
        # Encode graph structure via backbone
        h_node = self.backbone.encode(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        
        # Predict node mask (sigmoid to bound between 0 and 1)
        mask_logits = self.extractor(h_node)
        mask = torch.sigmoid(mask_logits)
        
        # Split into Causal and Environment subgraphs via node masking
        h_node_c = h_node * mask
        h_node_e = h_node * (1 - mask)
        
        # Pool to graph level
        h_graph_c = global_mean_pool(h_node_c, batch)
        h_graph_e = global_mean_pool(h_node_e, batch)
        
        # Project through bottlenecks
        h_shared_c = self.causal_bottleneck(h_graph_c)
        h_shared_e = self.env_bottleneck(h_graph_e)
        
        # Final predictions
        pred_c = self.causal_head(h_shared_c)
        pred_e = self.env_head(h_shared_e)
        
        return pred_c, pred_e, mask
