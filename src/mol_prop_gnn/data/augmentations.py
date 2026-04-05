"""Graph Augmentation Utilities for Contrastive Learning (GraphCL).

Provides transforms like random node dropping and edge masking to create
perturbed 'views' of molecular graphs.
"""

from __future__ import annotations

import random
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph


class GraphAugmentor:
    """Applies a random sequence of augmentations to a molecular graph.

    Parameters
    ----------
    node_drop_p : float
        Probability of dropping each node.
    edge_mask_p : float
        Probability of masking each edge's features.
    """

    def __init__(self, node_drop_p: float = 0.1, edge_mask_p: float = 0.1):
        self.node_drop_p = node_drop_p
        self.edge_mask_p = edge_mask_p

    def augment(self, data: Data) -> Data:
        """Apply a random augmentation to the data."""
        aug_data = data.clone()
        
        # We randomly pick ONE of the augmentations to apply per view
        # as suggested by the original GraphCL paper for best results.
        aug_type = random.choice(["node_drop", "edge_mask", "ident"])
        
        if aug_type == "node_drop":
            aug_data = self._random_node_drop(aug_data, self.node_drop_p)
        elif aug_type == "edge_mask":
            aug_data = self._random_edge_mask(aug_data, self.edge_mask_p)
            
        return aug_data

    def _random_node_drop(self, data: Data, p: float) -> Data:
        n_nodes = data.x.size(0)
        if n_nodes <= 1:
            return data
            
        keep_mask = torch.rand(n_nodes, device=data.x.device) > p
        if not keep_mask.any():
            keep_mask[random.randint(0, n_nodes - 1)] = True
            
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            keep_mask, 
            data.edge_index, 
            data.edge_attr, 
            relabel_nodes=True, 
            num_nodes=n_nodes
        )
        
        data.x = data.x[keep_mask]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        
        return data

    def _random_edge_mask(self, data: Data, p: float) -> Data:
        n_edges = data.edge_index.size(1)
        if n_edges == 0:
            return data
            
        # Mask edges by setting their features to zero or a small random perturbation
        # In molecular graphs, zeroing out features effectively 'masks' the bond type.
        mask = torch.rand(n_edges, device=data.edge_attr.device) < p
        data.edge_attr[mask] = 0.0
        
        return data


def augment_batch(batch: Batch, augmentor: GraphAugmentor) -> Batch:
    """Apply augmentations to an entire PyG Batch.
    
    Note: PyG Batch is a collection of disjoint graphs. Applying subgraphing 
    re-indexes them correctly within the large sparse adjacency matrix.
    """
    data_list = batch.to_data_list()
    aug_list = [augmentor.augment(d) for d in data_list]
    return Batch.from_data_list(aug_list)
