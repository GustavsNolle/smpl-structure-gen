"""Model factory for standardized backbone and multi-task model construction.

This ensures that both training and inference scripts can reconstruct 
the exact same architecture from saved hyperparameters.
"""

from __future__ import annotations
import logging
import torch.nn as nn

# Backbone imports
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.gin import MolGIN
from mol_prop_gnn.models.pna import MolPNA
from mol_prop_gnn.models.sage import MolGraphSAGE
from mol_prop_gnn.models.transformer import MolTransformerGNN
from mol_prop_gnn.models.joint_embedder import JointMolEmbedder
from mol_prop_gnn.models.causal_embedder import CausalMolEmbedder

logger = logging.getLogger(__name__)

def build_backbone(
    name: str, 
    node_dim: int, 
    edge_dim: int, 
    hidden_dim: int = 256, 
    layers: int = 5,
    deg: list[int] | None = None
) -> nn.Module:
    """Standardized backbone factory."""
    name = name.lower()
    if name == "gcn":
        return MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "gin":
        return MolGIN(node_input_dim=node_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "pna":
        if deg is None:
            raise ValueError("PNA backbone requires degree histogram 'deg'")
        return MolPNA(deg=deg, node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "sage":
        return MolGraphSAGE(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=layers)
    elif name == "transformer":
        return MolTransformerGNN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    else:
        raise ValueError(f"Unknown backbone: {name}")

def build_joint_model(
    backbone_name: str,
    node_dim: int,
    edge_dim: int,
    num_tasks: int,
    bottleneck_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 5,
    dropout: float = 0.3,
    deg: list[int] | None = None
) -> JointMolEmbedder:
    """Builds the full JointMolEmbedder with the specified backbone."""
    backbone = build_backbone(
        name=backbone_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        layers=num_layers,
        deg=deg
    )
    
    return JointMolEmbedder(
        backbone=backbone,
        backbone_out_dim=backbone.out_channels,
        num_datasets=num_tasks,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout
    )

def build_causal_model(
    backbone_name: str,
    node_dim: int,
    edge_dim: int,
    num_tasks: int,
    bottleneck_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 5,
    dropout: float = 0.3,
    deg: list[int] | None = None
) -> CausalMolEmbedder:
    """Builds the full CausalMolEmbedder with the specified backbone."""
    backbone = build_backbone(
        name=backbone_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        layers=num_layers,
        deg=deg
    )
    
    return CausalMolEmbedder(
        backbone=backbone,
        backbone_out_dim=backbone.out_channels,
        num_datasets=num_tasks,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout
    )
