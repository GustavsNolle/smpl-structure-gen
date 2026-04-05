"""Graph Autoencoder (GAE) for unsupervised node embedding extraction."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GAE


class GCNEncoder(nn.Module):
    """Encodes nodes into a latent space using GCN."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def create_gae(in_channels: int, latent_dim: int) -> GAE:
    """Creates a standard GAE with a GCN encoder."""
    return GAE(GCNEncoder(in_channels, latent_dim))
