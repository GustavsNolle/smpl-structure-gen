"""PretrainGINE — GINE encoder for self-supervised masked node prediction on ZINC.

Architecture follows OGB conventions with ZINC-specific adaptations:
- Atom/Bond type embeddings (discrete ZINC tokens → dense vectors)
- GINEConv blocks with LayerNorm (safe for batch size 1)
- OGB-style Virtual Node for global message passing
- Jumping Knowledge (concatenation of all layer outputs)
- global_add_pool for WL-test-level expressiveness
- Node-level prediction head (not graph-level) for masked atom classification

The [MASK] token is appended as index `num_atom_types` in the atom embedding table.
During pre-training, 15% of atoms have their type replaced with [MASK], and the
model learns to predict the original atom type from graph topology alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

# ── ZINC vocabulary constants ────────────────────────────────────────────
NUM_ATOM_TYPES = 28        # ZINC atom types: 0-27
MASK_TOKEN = NUM_ATOM_TYPES  # = 28, appended to embedding table
NUM_BOND_TYPES = 4         # ZINC bond types: 0-3


# ── Masking Transform ───────────────────────────────────────────────────

class AtomMaskTransform:
    """PyG Transform that randomly masks atom types for self-supervised learning.

    Applied dynamically (via ``transform=``, not ``pre_transform=``) so each
    epoch produces different masks.

    For each graph:
    1. Selects 15% of nodes uniformly at random
    2. Replaces their atom type with [MASK] token in ``data.x``
    3. Stores ground-truth labels in ``data.node_labels``
    4. Stores boolean mask in ``data.node_mask``

    Parameters
    ----------
    mask_rate : float
        Fraction of atoms to mask (default 0.15).
    num_atom_types : int
        Size of atom type vocabulary (default 28 for ZINC).
    """

    def __init__(self, mask_rate: float = 0.15, num_atom_types: int = NUM_ATOM_TYPES):
        self.mask_rate = mask_rate
        self.mask_token = num_atom_types  # Next index after vocabulary

    def __call__(self, data):
        num_nodes = data.x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_rate))

        # Store true atom types BEFORE masking
        data.node_labels = data.x.view(-1).clone().long()

        # Boolean mask: True for masked nodes
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        perm = torch.randperm(num_nodes)
        node_mask[perm[:num_mask]] = True
        data.node_mask = node_mask

        # Apply [MASK] token to selected nodes
        data.x = data.x.clone()
        if data.x.dim() == 2:
            data.x[node_mask, 0] = self.mask_token
        else:
            data.x[node_mask] = self.mask_token

        return data


# ── GINEBlock with LayerNorm ─────────────────────────────────────────────

class GINEPretrainBlock(nn.Module):
    """GINE message-passing block with LayerNorm (batch-size-1 safe).

    Uses GINEConv with edge_dim=None since bond types are pre-embedded
    to hidden_dim before entering the blocks. No double projection.
    """

    def __init__(self, hidden_dim: int, dropout: float, train_eps: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # edge_dim=None: expects edge_attr to already match node feature dim
        self.conv = GINEConv(nn=self.mlp, train_eps=train_eps)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# ── PretrainGINE Encoder ─────────────────────────────────────────────────

class PretrainGINE(nn.Module):
    """GINE encoder + node-level prediction head for masked atom prediction.

    This model is designed for Phase 1 self-supervised pre-training on ZINC.
    After training, the encoder weights (everything except ``prediction_head``)
    are saved and loaded as frozen/fine-tuned backbone in Phase 2.

    Parameters
    ----------
    num_atom_types : int
        Number of atom types in ZINC vocabulary (28). The embedding table has
        ``num_atom_types + 1`` entries to include the [MASK] token.
    num_bond_types : int
        Number of bond types in ZINC vocabulary (4).
    hidden_dim : int
        Hidden dimension for all layers.
    num_gnn_layers : int
        Number of GINE message-passing layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_atom_types: int = NUM_ATOM_TYPES,
        num_bond_types: int = NUM_BOND_TYPES,
        hidden_dim: int = 256,
        num_gnn_layers: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim

        # ── Embeddings ───────────────────────────────────────────────────
        # +1 for [MASK] token at index num_atom_types
        self.atom_embedding = nn.Embedding(num_atom_types + 1, hidden_dim)
        self.bond_embedding = nn.Embedding(num_bond_types, hidden_dim)

        # ── GNN blocks (LayerNorm) ───────────────────────────────────────
        self.blocks = nn.ModuleList([
            GINEPretrainBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        # ── Virtual Node (OGB-style, LayerNorm) ─────────────────────────
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight, 0)

        self.vn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_gnn_layers)
        ])

        # ── Jumping Knowledge output dim ─────────────────────────────────
        jk_dim = hidden_dim * (num_gnn_layers + 1)

        # ── Node-level prediction head (masked atom classification) ──────
        self.prediction_head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atom_types),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode atoms through GINE + Virtual Node + Jumping Knowledge.

        Returns per-node embeddings (NOT pooled) of shape ``(N, jk_dim)``.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        # Embed discrete atom and bond tokens → dense vectors
        h = self.atom_embedding(x.view(-1))       # (N, hidden_dim)
        e = self.bond_embedding(edge_attr.view(-1))  # (E, hidden_dim)

        h_list = [h]

        # Initialize virtual node (one per graph)
        vn_emb = self.vn_embedding(
            torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        )

        for i, block in enumerate(self.blocks):
            # Broadcast virtual node → all atoms
            h = h + vn_emb[batch]

            # GNN layer with residual connection
            h_prev = h
            h = block(h, edge_index, e)
            h = h + h_prev

            h_list.append(h)

            # Aggregate atoms → virtual node (with residual)
            vn_agg = global_add_pool(h, batch)
            vn_emb = self.vn_mlps[i](vn_agg + vn_emb)

        # Jumping Knowledge: concatenate all layer representations
        return torch.cat(h_list, dim=-1)  # (N, jk_dim)

    @property
    def out_channels(self) -> int:
        """Output dimension of the encoder (JK concatenation)."""
        return self.hidden_dim * (self.num_gnn_layers + 1)

    def forward(self, batch) -> torch.Tensor:
        """Full forward: encode → predict atom types for ALL nodes.

        Returns logits of shape ``(N, num_atom_types)`` — loss is computed
        externally only on masked nodes.
        """
        h = self.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.prediction_head(h)

    def get_encoder_state_dict(self) -> dict:
        """Extract only encoder weights (excludes prediction_head).

        This is what gets saved for Phase 2 fine-tuning.
        """
        full = self.state_dict()
        return {
            k: v for k, v in full.items()
            if not k.startswith("prediction_head.")
        }
