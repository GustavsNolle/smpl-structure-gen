"""GINE (Graph Isomorphism Network with Edge features) for molecular property prediction.

GINE extends GIN with state-of-the-art techniques from OGB and recent literature:

- Edge feature integration via PyG's GINEConv (single internal projection)
- Residual connections for gradient flow across deep networks
- Virtual Node (OGB standard) for global message passing without depth
- Jumping Knowledge: concatenates all layer outputs to combat over-smoothing
- GlobalAttention pooling: learns per-atom importance weights
- Global RDKit descriptor injection: compensates for GNN counting limitations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, global_add_pool


class GINEBlock(nn.Module):
    """A GINE message passing block with a 2-layer MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        dropout: float,
        train_eps: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        # PyG's GINEConv creates its own internal projection for edge_dim -> in_dim
        self.conv = GINEConv(nn=self.mlp, train_eps=train_eps, edge_dim=edge_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MolGINE(nn.Module):
    """GINE encoder with Virtual Node, Jumping Knowledge, GlobalAttention pooling,
    and global RDKit descriptor injection.

    Parameters
    ----------
    node_input_dim : int
        Dimension of atom features.
    edge_input_dim : int
        Dimension of bond features (passed raw to PyG's internal projection).
    hidden_dim : int
        Hidden dimension for GINE layers.
    num_gnn_layers : int
        Number of message-passing layers.
    decoder_hidden_dim : int
        Hidden dimension in the readout MLP.
    dropout : float
        Dropout probability.
    output_dim : int
        Number of output tasks.
    global_features_dim : int
        Dimension of global molecular descriptors (RDKit). Set to 0 to disable.
    """

    def __init__(
        self,
        node_input_dim: int = 38,
        edge_input_dim: int = 13,
        hidden_dim: int = 256,
        num_gnn_layers: int = 5,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.3,
        output_dim: int = 1,
        global_features_dim: int = 10,
    ) -> None:
        super().__init__()

        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim
        self.global_features_dim = global_features_dim

        # ── Node projection ──────────────────────────────────────────────
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # ── GNN blocks ───────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            GINEBlock(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=edge_input_dim,  # raw edge dim — PyG projects internally
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])

        # ── Virtual Node (OGB-style) ────────────────────────────────────
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight, 0)

        self.vn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_gnn_layers)
        ])

        # ── Jumping Knowledge output dim ────────────────────────────────
        jk_dim = hidden_dim * (num_gnn_layers + 1)

        # ── Global Attention Pooling ────────────────────────────────────
        gate_nn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = GlobalAttention(gate_nn)

        # ── Readout MLP (JK dim + global molecular descriptors) ─────────
        readout_input_dim = jk_dim + global_features_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode atom features with GINE + Virtual Node + Jumping Knowledge."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        h = self.node_proj(x)
        h_list = [h]

        # Initialize virtual node embedding (one per graph in batch)
        vn_emb = self.vn_embedding(
            torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        )

        for i, block in enumerate(self.blocks):
            # Broadcast virtual node -> all atoms
            h = h + vn_emb[batch]

            # GNN layer with residual connection
            h_prev = h
            h = block(h, edge_index, edge_attr)
            h = h + h_prev

            h_list.append(h)

            # Aggregate atoms -> virtual node (with residual)
            vn_agg = global_add_pool(h, batch)
            vn_emb = self.vn_mlps[i](vn_agg + vn_emb)

        # Jumping Knowledge: concatenate all layer representations
        return torch.cat(h_list, dim=-1)

    @property
    def out_channels(self) -> int:
        return self.hidden_dim * (self.num_gnn_layers + 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr, batch)
        h_pooled = self.pool(h, batch)

        # Inject global RDKit descriptors
        if global_features is not None:
            if global_features.dim() == 3:
                global_features = global_features.squeeze(1)
            h_pooled = torch.cat([h_pooled, global_features], dim=-1)
        elif self.global_features_dim > 0:
            # Zero-pad if descriptors not provided (graceful fallback)
            h_pooled = torch.cat([
                h_pooled,
                torch.zeros(h_pooled.size(0), self.global_features_dim, device=h_pooled.device),
            ], dim=-1)

        return self.readout(h_pooled)
