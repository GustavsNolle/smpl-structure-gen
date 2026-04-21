import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization initialized to zero (AdaLN-Zero).
    
    Generates Scale (gamma) and Shift (beta) parameters from the condition vector.
    The final linear layer is initialized to zero so it acts as an identity function
    at initialization.
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        
        # Zero-Init final layer
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [N, hidden_dim]
            cond: Conditioning tensor [N, cond_dim]
        Returns:
            Modulated tensor [N, hidden_dim]
        """
        # emb: [N, 2 * hidden_dim]
        emb = self.linear(self.silu(cond))
        scale, shift = emb.chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class DiGressTransformerBlock(nn.Module):
    """A single Graph Transformer block with AdaLN-Zero conditioning."""
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_ln_1 = AdaLNZero(cond_dim, hidden_dim)
        
        # Using PyG's TransformerConv for message passing.
        # It natively handles edge attributes.
        self.attn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=hidden_dim,
            concat=True,
            dropout=0.1
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_ln_2 = AdaLNZero(cond_dim, hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Edge Updater to allow edge features to evolve
        self.edge_updater = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        cond_node: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_total_nodes, hidden_dim]
            edge_index: [2, num_total_edges]
            edge_attr: Edge features [num_total_edges, hidden_dim]
            cond_node: Condition vector broadcasted to nodes [num_total_nodes, cond_dim]
        Returns:
            Updated node features [num_total_nodes, hidden_dim]
        """
        # Pre-LN + AdaLN for Attention
        h = self.norm1(x)
        h = self.ada_ln_1(h, cond_node)
        
        # MSA
        h = self.attn(x=h, edge_index=edge_index, edge_attr=edge_attr)
        x = x + h
        
        # Pre-LN + AdaLN for FFN
        h = self.norm2(x)
        h = self.ada_ln_2(h, cond_node)
        
        # FFN
        h = self.ffn(h)
        x = x + h
        
        # Update edges based on contextualized nodes (Symmetric)
        src, dst = edge_index
        edge_updated = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_updater(edge_updated)
        
        return x, edge_attr


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Generates sinusoidal embeddings for timesteps.
    
    Args:
        timesteps: [batch_size]
        dim: Embedding dimension
    Returns:
        [batch_size, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ConditionalDiGressNet(nn.Module):
    """Conditional Discrete Denoising Diffusion Model backbone.
    
    Predicts categorical logits for nodes and edges to drive Markov transition matrices.
    Supports Classifier-Free Guidance (CFG).
    """
    def __init__(
        self, 
        num_node_classes: int, 
        num_edge_classes: int, 
        hidden_dim: int = 128, 
        causal_cond_dim: int = 128,
        num_layers: int = 5,
        num_heads: int = 4
    ):
        super().__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.hidden_dim = hidden_dim
        self.causal_cond_dim = causal_cond_dim
        
        # Embeddings for discrete classes
        self.node_emb = nn.Embedding(num_node_classes, hidden_dim)
        self.edge_emb = nn.Embedding(num_edge_classes, hidden_dim)
        
        # Structural Encodings
        self.num_cycle_types = 4  # 3, 4, 5, 6-membered rings
        self.num_laplace_dims = 8
        self.struct_emb = nn.Linear(self.num_cycle_types + self.num_laplace_dims, hidden_dim)
        
        # Timestep embedding dimension
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_dim * 2, self.time_dim)
        )
        
        # The condition_vec is the concatenation of time_emb and causal_emb
        self.cond_dim = self.time_dim + causal_cond_dim
        
        # Learnable null embedding for CFG dropout
        self.null_embedding = nn.Parameter(torch.randn(1, causal_cond_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiGressTransformerBlock(hidden_dim, self.cond_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final output heads predicting categorical logits
        self.node_pred_head = nn.Linear(hidden_dim, num_node_classes)
        # Symmetrized input: node_sum + edge_features -> hidden_dim + hidden_dim = 2 * hidden_dim
        self.edge_pred_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_edge_classes)
        )

    def forward(
        self, 
        X: torch.Tensor, 
        E: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch_idx: torch.Tensor,
        t: torch.Tensor, 
        c: torch.Tensor,
        p_uncond: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with CFG logic.
        
        Args:
            X: Noisy node categorical indices [num_total_nodes]
            E: Noisy edge categorical indices [num_total_edges]
            edge_index: Edge indices [2, num_total_edges]
            batch_idx: Batch assignment for nodes [num_total_nodes]
            t: Timesteps [batch_size]
            c: Causal embeddings [batch_size, causal_cond_dim]
            p_uncond: Probability of replacing causal embedding with null_embedding.
        Returns:
            Tuple of (node_logits, edge_logits)
        """
        batch_size = t.shape[0]
        
        # CFG Dropout: Generate boolean mask for unconditional generation
        if self.training and p_uncond > 0.0:
            mask = torch.rand(batch_size, 1, device=c.device) < p_uncond
            # Replace with null_embedding where mask is True
            c = torch.where(mask, self.null_embedding.expand(batch_size, -1), c)
            
        return self._forward_impl(X, E, edge_index, batch_idx, t, c)

    def _forward_impl(
        self, 
        X: torch.Tensor, 
        E: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch_idx: torch.Tensor,
        t: torch.Tensor, 
        c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core forward logic without CFG mask application."""
        # 1. Structural Encodings Calculation
        # We need the adjacency matrix for structural features
        # Mapping edge types to weights: No Bond=0, others=1.0 for simplicity in topology
        # Alternatively, use bond orders: 1, 2, 3, 1.5
        edge_weights = torch.zeros_like(E, dtype=torch.float32)
        edge_weights[E == 1] = 1.0 # Single
        edge_weights[E == 2] = 2.0 # Double
        edge_weights[E == 3] = 3.0 # Triple
        edge_weights[E == 4] = 1.5 # Aromatic
        
        # [batch_size, max_nodes, max_nodes]
        adj = to_dense_adj(edge_index, batch_idx, edge_weights)
        device = X.device
        batch_size = adj.shape[0]
        max_nodes = adj.shape[1]
        
        # A. Cycle Features (diag(A^k) for k=3,4,5,6)
        cycle_features = []
        curr_adj = adj
        for k in [2, 3, 4, 5]: # We want A^3, A^4, A^5, A^6. Wait, the loop starts at A^1.
            # We'll just do it explicitly for clarity
            pass
            
        A2 = torch.bmm(adj, adj)
        A3 = torch.bmm(A2, adj)
        A4 = torch.bmm(A3, adj)
        A5 = torch.bmm(A4, adj)
        A6 = torch.bmm(A5, adj)
        
        c3 = torch.diagonal(A3, dim1=-2, dim2=-1)
        c4 = torch.diagonal(A4, dim1=-2, dim2=-1)
        c5 = torch.diagonal(A5, dim1=-2, dim2=-1)
        c6 = torch.diagonal(A6, dim1=-2, dim2=-1)
        
        # [batch_size, max_nodes, 4]
        cycles = torch.stack([c3, c4, c5, c6], dim=-1)
        # Normalize to avoid exploding values in dense graphs
        cycles = torch.log1p(cycles)
        
        # B. Laplacian Eigenvectors
        # L = D - A. D is diagonal sum of rows.
        D = torch.diag_embed(adj.sum(dim=-1))
        L = D - adj
        
        # Compute eigenvalues/vectors
        # Using eigh for symmetric matrices
        try:
            evals, evecs = torch.linalg.eigh(L)
            # Take first num_laplace_dims eigenvectors (ignoring the first one which is constant for connected graphs)
            # evecs: [batch_size, max_nodes, max_nodes]
            # We take indices 1 to num_laplace_dims+1
            laplace_feats = evecs[:, :, 1:1 + self.num_laplace_dims]
            # Zero pad if max_nodes is smaller than requested dims
            if laplace_feats.shape[2] < self.num_laplace_dims:
                padding = self.num_laplace_dims - laplace_feats.shape[2]
                laplace_feats = F.pad(laplace_feats, (0, padding))
        except Exception:
            # Fallback for non-convergent cases or singular matrices in noise
            laplace_feats = torch.zeros((batch_size, max_nodes, self.num_laplace_dims), device=device)
            
        # Combine structural features
        struct_feats_dense = torch.cat([cycles, laplace_feats], dim=-1)
        
        # Flatten back to sparse node representation
        # [num_total_nodes, struct_dim]
        # We need a mask to extract the actual nodes from the dense batch
        _, counts = torch.unique(batch_idx, return_counts=True)
        struct_feats_list = []
        for i, count in enumerate(counts):
            struct_feats_list.append(struct_feats_dense[i, :count])
        struct_feats = torch.cat(struct_feats_list, dim=0)
        
        # 2. Embeddings
        x_emb = self.node_emb(X)  # [num_total_nodes, hidden_dim]
        e_emb = self.edge_emb(E)  # [num_total_edges, hidden_dim]
        
        # Inject structural information
        x_emb = x_emb + self.struct_emb(struct_feats)
        
        # 2. Timestep + Embedding Fusion
        # t: [batch_size] -> t_emb: [batch_size, time_dim]
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # condition_vec: [batch_size, cond_dim]
        condition_vec = torch.cat([t_emb, c], dim=-1)
        
        # Broadcast condition_vec to nodes
        # cond_node: [num_total_nodes, cond_dim]
        cond_node = condition_vec[batch_idx]
        
        # 3. Message Passing with AdaLN-Zero
        for block in self.blocks:
            x_emb, e_emb = block(x_emb, edge_index, e_emb, cond_node)
            
        # 4. Predict Logits
        node_logits = self.node_pred_head(x_emb)  # [num_total_nodes, num_node_classes]
        
        # Symmetrized edge prediction: node_sum + edge_features
        src, dst = edge_index
        # node_pair_repr is permutation-invariant (undirected)
        node_pair_repr = x_emb[src] + x_emb[dst]
        edge_repr = torch.cat([node_pair_repr, e_emb], dim=-1)
        edge_logits = self.edge_pred_head(edge_repr)  # [num_total_edges, num_edge_classes]
        
        return node_logits, edge_logits

    @torch.no_grad()
    def predict_cfg_logits(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        guidance_scale: float = 2.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CFG Sampling Logic (The Reverse Process).
        
        Runs the network twice and applies CFG math to the raw unnormalized logits.
        
        Args:
            X, E, edge_index, batch_idx, t: Graph state at timestep t.
            c: Target causal embeddings [batch_size, causal_cond_dim]
            guidance_scale (w): Strength of guidance.
        Returns:
            final_node_logits, final_edge_logits
        """
        batch_size = t.shape[0]
        
        # 1. Unconditional pass (logits_unc)
        c_uncond = self.null_embedding.expand(batch_size, -1)
        node_logits_unc, edge_logits_unc = self._forward_impl(
            X, E, edge_index, batch_idx, t, c_uncond
        )
        
        # 2. Conditional pass (logits_cond)
        node_logits_cond, edge_logits_cond = self._forward_impl(
            X, E, edge_index, batch_idx, t, c
        )
        
        # 3. CFG Math: Extrapolate unnormalized logits
        # final_logits = logits_unc + w * (logits_cond - logits_unc)
        final_node_logits = node_logits_unc + guidance_scale * (node_logits_cond - node_logits_unc)
        final_edge_logits = edge_logits_unc + guidance_scale * (edge_logits_cond - edge_logits_unc)
        
        return final_node_logits, final_edge_logits
