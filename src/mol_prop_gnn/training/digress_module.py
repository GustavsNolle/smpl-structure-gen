import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path

from clearml import Model
from mol_prop_gnn.models.digress import ConditionalDiGressNet
from mol_prop_gnn.models.factory import build_causal_model
from mol_prop_gnn.models.mini_judge_gin import MiniJudgeGIN
from torch_geometric.utils import to_dense_adj
from mol_prop_gnn.data.preprocessing import get_node_feature_dim, get_edge_feature_dim

logger = logging.getLogger(__name__)


def get_noise_schedule(num_timesteps: int, device: torch.device, s: float = 0.008):
    """Cosine schedule as proposed in improved DDPM / DiGress."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32, device=device)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Extract beta_t to ensure it doesn't exceed 0.999
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.999)
    
    # Recompute alpha_bar
    alphas = 1.0 - betas
    alpha_bar_t = torch.cumprod(alphas, dim=0)
    return alpha_bar_t

def add_marginal_noise(
    x: torch.Tensor, 
    alpha_bar_t: torch.Tensor, 
    marginals: torch.Tensor
) -> torch.Tensor:
    """Adds discrete noise biased towards the marginal distribution.
    
    P(X_t | X_0) = alpha_bar_t * I + (1 - alpha_bar_t) * 1 * m^T
    """
    num_classes = marginals.shape[0]
    batch_size = x.shape[0]
    
    # Probabilities: alpha_bar_t if staying in X_0, plus (1-alpha_bar_t) * marginal
    # probs[i, j] = (1 - alpha_bar_t[i]) * marginals[j]
    # If j == x[i], add alpha_bar_t[i]
    
    # Broadcast marginals to [batch_size, num_classes]
    m = marginals.view(1, -1).expand(batch_size, -1)
    
    # Probability of transitioning to any state j
    probs = (1.0 - alpha_bar_t.view(-1, 1)) * m
    
    # Add alpha_bar_t to the probability of staying in the true state
    probs.scatter_add_(1, x.unsqueeze(1), alpha_bar_t.view(-1, 1))
    
    # Sample noisy state
    # Clamp for numerical stability on GPU (prevent zero-sum rows)
    probs = torch.clamp(probs, min=1e-10)
    noisy_x = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return noisy_x

def denseify_graphs(batch_edge_index, batch_edge_attr, batch_idx, num_edge_classes):
    """Converts sparse edge indices to fully connected (without self-loops).
    Assigns the 'no bond' class to non-edges using a robust matrix-based mapping.
    """
    device = batch_idx.device
    batch_size = batch_idx.max().item() + 1
    _, counts = torch.unique(batch_idx, return_counts=True)
    
    offset = 0
    edge_index_list = []
    E_dense_list = []
    
    for i, c_nodes in enumerate(counts):
        # 1. Create a dense bond-type matrix for this graph
        # Initialize as 'no bond' (class 0)
        E_matrix = torch.zeros((c_nodes, c_nodes), dtype=torch.long, device=device)
        
        # 2. Map existing sparse edges into the matrix
        graph_edge_mask = batch_idx[batch_edge_index[0]] == i
        if graph_edge_mask.any():
            g_edge_index = batch_edge_index[:, graph_edge_mask] - offset
            g_edge_attr = batch_edge_attr[graph_edge_mask]
            
            # Map one-hot bond attributes to categorical integers (1-5)
            # Indices: 0:SINGLE, 1:DOUBLE, 2:TRIPLE, 3:AROMATIC, 4:OTHER
            existing_e_types = torch.argmax(g_edge_attr[:, :5], dim=1) + 1
            E_matrix[g_edge_index[0], g_edge_index[1]] = existing_e_types
            
        # 3. Extract the dense edges (excluding self-loops)
        # Create a mask for off-diagonal elements
        mask = ~torch.eye(c_nodes, dtype=torch.bool, device=device)
        
        # Get indices for off-diagonal elements
        row_idx, col_idx = torch.where(mask)
        E_true_g = E_matrix[mask]
        
        # 4. Global offset for the batch
        edge_index_list.append(torch.stack([row_idx + offset, col_idx + offset], dim=0))
        E_dense_list.append(E_true_g)
        
        offset += c_nodes
        
    return torch.cat(edge_index_list, dim=1), torch.cat(E_dense_list, dim=0)


def digress_to_judge_features(X, E, edge_index, num_node_classes=11, num_edge_classes=6):
    """Converts DiGress categorical atom/bond types to Judge featurization.
    
    DiGress (11 classes): 0:C, 1:N, 2:O, 3:F, 4:P, 5:S, 6:Cl, 7:Br, 8:I, 9-10:Other
    Judge (38 dims): One-hot atom types (C, N, O, S, F, P, Cl, Br, I, ...) + structural info.
    
    Since structural info (degree, formal charge) is not explicitly tracked in DiGress's
    discrete state, we provide the atom-type one-hot and zero out the rest.
    """
    device = X.device
    num_nodes = X.shape[0]
    num_edges = E.shape[0]
    
    # 1. Node Features (38 dims)
    x_rdkit = torch.zeros((num_nodes, 38), device=device)
    for i in range(9):
        mask = (X == i)
        x_rdkit[mask, i] = 1.0
    mask_other = (X >= 9)
    x_rdkit[mask_other, 9] = 1.0
    
    # 2. Edge Features (12 dims)
    e_rdkit = torch.zeros((num_edges, 12), device=device)
    for i in range(1, 6):
        mask = (E == i)
        e_rdkit[mask, i-1] = 1.0
        
    return x_rdkit, e_rdkit

class DiGressModule(pl.LightningModule):
    """PyTorch Lightning Module for training Conditional DiGress."""

    def __init__(
        self,
        num_node_classes: int = 11,  # 10 Atom types + 1 other
        num_edge_classes: int = 6,   # 5 Bond types + 1 other/none
        hidden_dim: int = 128,
        causal_cond_dim: int = 128,
        num_layers: int = 5,
        num_heads: int = 4,
        num_timesteps: int = 1000,
        learning_rate: float = 1e-3,
        causal_judge_model_id: str = "94f148c657ed4b7b8fdaa54b4ad2bdd3",
        property_judge_ids: dict = None,
        p_uncond: float = 0.15
    ):

        super().__init__()
        self.save_hyperparameters()
        
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.p_uncond = p_uncond
        
        # Instantiate the DiGress generator backbone
        self.model = ConditionalDiGressNet(
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            hidden_dim=hidden_dim,
            causal_cond_dim=causal_cond_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # We will load the frozen judges dynamically in setup()
        self.causal_judge = None
        self.causal_judge_model_id = causal_judge_model_id
        
        # New Mini Judges for specialized property guidance
        self.property_judges = nn.ModuleDict()
        self.property_judge_ids = {} # Populated from config
        
        # Load ZINC marginals
        marginals_path = Path("data/zinc_marginals.pt")
        
        # Hardcoded Dense ZINC Edge Marginals (Idealized Sparsity)
        # 95% No-Bond, 4% Single, 0.5% Double, 0.2% Triple, 0.2% Aromatic, 0.1% Other
        edge_m = torch.tensor([0.95, 0.04, 0.005, 0.002, 0.002, 0.001])
        self.register_buffer("edge_marginals", edge_m)
        
        # Node Marginals: load from file or use uniform
        if marginals_path.exists():
            logger.info("Loading ZINC node marginals...")
            marginals = torch.load(marginals_path)
            node_m = marginals["node_marginals"]
        else:
            logger.warning("ZINC marginals not found! Falling back to uniform for nodes.")
            node_m = torch.ones(num_node_classes) / num_node_classes
        
        # CRITICAL: Floor all marginals to prevent zero-probability classes.
        # Zero entries cause NaN in multinomial sampling at high noise levels.
        node_m = torch.clamp(node_m, min=1e-6)
        node_m = node_m / node_m.sum()  # Re-normalize
        self.register_buffer("node_marginals", node_m)

        # GPU Bottleneck Fix: Register noise schedule as buffer
        alpha_bar_all = get_noise_schedule(num_timesteps, torch.device('cpu'))
        self.register_buffer("alpha_bar_all", alpha_bar_all)

    def setup(self, stage=None):
        """Loads the pre-trained frozen judges from ClearML."""
        # 1. Load Original Causal Judge
        if self.causal_judge is None:
            logger.info(f"Loading frozen Causal Judge (ClearML ID: {self.causal_judge_model_id})...")
            model_info = Model(model_id=self.causal_judge_model_id)
            local_path = model_info.get_local_copy()
            ckpt = torch.load(local_path, map_location=self.device, weights_only=False)
            model_config = ckpt.get("hyper_parameters", {}).get("model_config", {})
            
            judge_model = build_causal_model(
                backbone_name=model_config.get("backbone_name", "gin"),
                node_dim=get_node_feature_dim(),
                edge_dim=get_edge_feature_dim(),
                num_tasks=model_config.get("num_tasks", 21),
                bottleneck_dim=model_config.get("bottleneck_dim", 256),
                hidden_dim=model_config.get("hidden_dim", 256),
                num_layers=model_config.get("num_layers", 5),
                dropout=model_config.get("dropout", 0.3)
            )
            state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            judge_model.load_state_dict(state_dict)
            self.causal_judge = judge_model.to(self.device).eval()
            for p in self.causal_judge.parameters(): p.requires_grad = False
            logger.info("Causal Judge loaded successfully.")

        # 2. Load Mini Property Judges
        if not self.property_judges and hasattr(self.hparams, 'property_judge_ids'):
            ids = self.hparams.property_judge_ids
            if isinstance(ids, dict):
                for prop, mid in ids.items():
                    logger.info(f"Loading Mini Judge for {prop} (ClearML ID: {mid})...")
                    m_info = Model(model_id=mid)
                    l_path = m_info.get_local_copy()
                    m_ckpt = torch.load(l_path, map_location=self.device, weights_only=False)
                    mh = m_ckpt.get("hyper_parameters", {})
                    
                    p_model = MiniJudgeGIN(
                        node_input_dim=38, # Standard RDKit featurization
                        hidden_dim=mh.get("hidden_dim", 64),
                        num_layers=mh.get("num_layers", 4),
                        dropout=mh.get("dropout", 0.1)
                    )
                    # Load state dict (MiniJudgeModule has self.model)
                    s_dict = {k[6:]: v for k, v in m_ckpt["state_dict"].items() if k.startswith("model.")}
                    p_model.load_state_dict(s_dict)
                    p_model = p_model.to(self.device).eval()
                    for p in p_model.parameters(): p.requires_grad = False
                    self.property_judges[prop] = p_model
                logger.info(f"Loaded {len(self.property_judges)} Mini Judges.")

    def forward(self, batch):
        # We don't typically use the forward method directly in diffusion modules.
        pass

    def get_causal_embedding(self, batch) -> torch.Tensor:
        """Extracts combined embeddings from all judges.
        
        Concatenates:
        - 256-dim Causal Subgraph embedding from original judge.
        - 3x64-dim Graph-level embeddings from mini judges (LogP, QED, SAS).
        Total: 448 dims.
        """
        from torch_geometric.nn import global_mean_pool
        
        # 1. Prepare features (DiGress -> Judge featurization)
        # Note: If batch already has RDKit features (e.g. from data loader), use them.
        # But during sampling, we only have atom/edge types, so we must convert.
        if batch.x.shape[1] == self.num_node_classes:
            x_j, e_j = digress_to_judge_features(batch.x.argmax(dim=-1), 
                                                batch.edge_attr.argmax(dim=-1), 
                                                batch.edge_index)
        else:
            x_j, e_j = batch.x, batch.edge_attr

        embeddings = []

        # 2. Extract from Causal Judge
        if self.causal_judge is not None:
            self.causal_judge.eval()
            with torch.no_grad():
                h_node = self.causal_judge.backbone.encode(x=x_j, edge_index=batch.edge_index, 
                                                        edge_attr=e_j, batch=batch.batch)
                mask = torch.sigmoid(self.causal_judge.extractor(h_node))
                h_graph_c = global_mean_pool(h_node * mask, batch.batch)
                causal_emb = self.causal_judge.causal_bottleneck(h_graph_c)
                embeddings.append(causal_emb)

        # 3. Extract from Mini Judges
        for prop in ["logp", "qed", "sascore"]:
            if prop in self.property_judges:
                p_judge = self.property_judges[prop]
                p_judge.eval()
                with torch.no_grad():
                    # We need the graph-level latent before the scalar head
                    # h_graph is the output of global_mean_pool in MiniJudgeGIN
                    # To avoid modifying MiniJudgeGIN, we'll re-implement the pooling here
                    # using the model's internal layers if possible, but it's cleaner
                    # to just get the final scalar if we want scalar conditioning,
                    # OR we could modify MiniJudgeGIN to return both.
                    # Given the 192-dim target, the user likely wants the Laten space.
                    
                    # Re-running the logic to get the latent (h_graph from line 119 in mini_judge_gin.py)
                    # Let's add a helper to MiniJudgeGIN or just use the scalar for now?
                    # No, 3x64 = 192. So we need the latent.
                    
                    # HACK: If we can't change the model, we'll just project the scalar? 
                    # No, that's bad. I'll assume MiniJudgeGIN.forward can return the latent.
                    # Wait, I'll just use the internal layers since it's a Sequential/ModuleList.
                    
                    h = p_judge.node_proj(x_j)
                    num_graphs = batch.batch.max().item() + 1
                    vn_emb = torch.zeros(num_graphs, p_judge.hidden_dim, device=x_j.device)
                    for layer_idx, block in enumerate(p_judge.blocks):
                        h = h + vn_emb[batch.batch]
                        h = block(h, batch.edge_index)
                        h_pool = global_mean_pool(h, batch.batch)
                        vn_emb = vn_emb + p_judge.vn_encoder[layer_idx](h_pool)
                    
                    h = p_judge.final_norm(h)
                    h_graph = global_mean_pool(h, batch.batch)
                    embeddings.append(h_graph)

        if not embeddings:
            return torch.zeros((batch.num_graphs, self.hparams.causal_cond_dim), device=self.device)
            
        return torch.cat(embeddings, dim=-1)

    def _shared_step(self, batch, stage: str, p_uncond: float):
        # 1. Extract Target Causal Embedding (Condition)
        c = self.get_causal_embedding(batch)  # [batch_size, causal_cond_dim]
        
        batch_size = c.shape[0]
        
        # 2. Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # 3. Convert RDKit continuous/one-hot features to categorical integers
        if hasattr(batch, 'node_type'):
            X_true = batch.node_type
        else:
            X_true = torch.argmax(batch.x[:, :10], dim=1) % self.num_node_classes
        
        # Defense-in-depth: clamp indices to valid ranges to prevent CUDA asserts
        X_true = torch.clamp(X_true, 0, self.num_node_classes - 1)
            
        # 3b. Denseify graphs for discrete diffusion (fully connected topology)
        edge_index_dense, E_true = denseify_graphs(
            batch_edge_index=batch.edge_index,
            batch_edge_attr=batch.edge_attr,
            batch_idx=batch.batch,
            num_edge_classes=self.num_edge_classes
        )
        E_true = torch.clamp(E_true, 0, self.num_edge_classes - 1)
            
        # 4. Forward Diffusion (Add Noise)
        alpha_bar_t = self.alpha_bar_all[t]
        
        # Broadcast alpha_bar_t to nodes and edges
        alpha_bar_t_nodes = alpha_bar_t[batch.batch]
        alpha_bar_t_edges = alpha_bar_t[batch.batch[edge_index_dense[0]]]
        
        X_noisy = add_marginal_noise(X_true, alpha_bar_t_nodes, self.node_marginals)
        E_noisy = add_marginal_noise(E_true, alpha_bar_t_edges, self.edge_marginals)
        
        # 5. Reverse Diffusion (Predict Denoised Logits)
        node_logits, edge_logits = self.model(
            X=X_noisy,
            E=E_noisy,
            edge_index=edge_index_dense,
            batch_idx=batch.batch,
            t=t,
            c=c,
            p_uncond=p_uncond
        )
        
        # 6. Loss Calculation (Cross-Entropy)
        # Node weights: Rare Element Boost to prevent Carbon-bias
        # Indices: 0:C, 1:N, 2:O, 3:F, 4:P, 5:S, 6:Cl, 7:Br, 8:I, 9:Other, 10:Other
        node_loss_weights = torch.tensor(
            [1.0, 1.5, 1.5, 3.0, 5.0, 5.0, 3.0, 4.0, 5.0, 2.0, 2.0], 
            device=self.device
        )
        loss_nodes = F.cross_entropy(node_logits, X_true, weight=node_loss_weights)
        
        # Weighted Edge Loss: Class 0 (No Bond) is 95% of data. 
        # We weight bond classes (1-5) higher to force structural learning.
        edge_loss_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 5.0], device=self.device)
        loss_edges = F.cross_entropy(edge_logits, E_true, weight=edge_loss_weights)
        
        # 7. Auxiliary Structural Losses
        # A. Expected Adjacency from denoised logits
        edge_probs = F.softmax(edge_logits, dim=-1)
        # Bond orders: [0: No Bond, 1: Single, 2: Double, 3: Triple, 4: Aromatic, 5: Other]
        bond_orders = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5, 1.0], device=self.device)
        expected_bonds = (edge_probs * bond_orders.unsqueeze(0)).sum(dim=-1) # [Total_Edges]

        # Convert back to dense batch for spectral calculations
        adj_dense = to_dense_adj(edge_index_dense, batch.batch, expected_bonds)
        
        # FIX 1: Force perfect symmetry to prevent NaN gradients in eigvalsh
        # Attention predictions for u->v and v->u might differ slightly during training
        adj_dense = (adj_dense + adj_dense.transpose(1, 2)) / 2.0
        
        batch_size, max_nodes, _ = adj_dense.shape
        
        # B. Laplacian Connectivity Loss (Anti-Fragment)
        # CRITICAL: eigvalsh backward is numerically unstable for near-singular
        # matrices (common in early training). We compute the Fiedler value
        # WITHOUT gradients and create a differentiable proxy via the adjacency.
        try:
            with torch.no_grad():
                D_dense = torch.diag_embed(adj_dense.sum(dim=-1))
                L_dense = D_dense - adj_dense
                evals_dense = torch.linalg.eigvalsh(L_dense)
            if evals_dense.shape[1] > 1:
                lambda2 = evals_dense[:, 1]
                # Use the detached eigenvalue as a TARGET for a differentiable proxy:
                # Penalize low connectivity via the minimum row-sum of adj (proxy for Fiedler)
                min_degree = adj_dense.sum(dim=-1).min(dim=-1).values
                loss_conn = F.relu(0.1 - min_degree).mean()
            else:
                loss_conn = torch.tensor(0.0, device=self.device)
        except Exception:
            loss_conn = torch.tensor(0.0, device=self.device)
        
        # C. Density Prior (Anti-Hairball)
        # Instead of recalculating noisy eigenvectors, we penalize the model
        # for predicting too many dense connections relative to a linear graph.
        # A realistic drug has roughly num_nodes to num_nodes + 4 edges.
        
        # Expected total edges per graph (divide by 2 because it's bidirectional)
        expected_total_edges = adj_dense.sum(dim=(1, 2)) / 2.0 
        
        # Number of nodes per graph
        # Using bincount on batch.batch to get [batch_size]
        nodes_per_graph = torch.bincount(batch.batch)
        
        # Penalize if the model tries to draw more edges than a fused 3-ring system would allow 
        max_allowed_edges = nodes_per_graph.float() + 3.0
        loss_dist = F.relu(expected_total_edges - max_allowed_edges).mean()

        # D. Ring Complexity Loss (Anti-Strained + Anti-Macrocycle)
        # Tr(A^k) counts the number of closed walks of length k. For a simple graph:
        #   - Tr(A^3)/6 ≈ number of triangles (epoxide-like strained 3-rings)
        #   - Tr(A^4)/8 ≈ number of cyclobutanes (strained 4-rings)
        # Drug-like molecules (SA score < 3) rarely have these features.
        
        # Binary adjacency for ring counting (detached — no gradients needed for penalty)
        adj_binary = (adj_dense.detach() > 0.3).float()
        adj_binary[:, range(max_nodes), range(max_nodes)] = 0  # Remove self-loops (batched)
        
        # Compute powers efficiently: reuse intermediate results
        A2 = torch.bmm(adj_binary, adj_binary)
        A3 = torch.bmm(A2, adj_binary)
        A4 = torch.bmm(A3, adj_binary)
        
        trace_A3 = torch.diagonal(A3, dim1=-2, dim2=-1).sum(dim=-1)
        trace_A4 = torch.diagonal(A4, dim1=-2, dim2=-1).sum(dim=-1)
        
        # Large ring penalties: compute A^9 and A^10 efficiently
        # A^8 = A^4 @ A^4 (reuse!), A^9 = A^8 @ A, A^10 = A^8 @ A^2 (reuse!)
        # This is 3 bmm calls instead of 6
        A8 = torch.bmm(A4, A4)
        A9 = torch.bmm(A8, adj_binary)
        A10 = torch.bmm(A8, A2)
        
        trace_A9 = torch.diagonal(A9, dim1=-2, dim2=-1).sum(dim=-1)
        trace_A10 = torch.diagonal(A10, dim1=-2, dim2=-1).sum(dim=-1)
        
        n_nodes_f = nodes_per_graph.float().clamp(min=1.0)
        
        small_ring_penalty = (
            F.relu(trace_A3 / n_nodes_f - 1.5) +
            F.relu(trace_A4 / n_nodes_f - 2.0)
        ).mean()
        
        macro_ring_penalty = (
            F.relu(trace_A9 / n_nodes_f - 80.0) +
            F.relu(trace_A10 / n_nodes_f - 150.0)
        ).mean()
        
        loss_ring = small_ring_penalty + macro_ring_penalty


        # E. Valency Penalty
        src, dst = edge_index_dense
        node_valency = torch.zeros(X_noisy.shape[0], device=self.device)
        node_valency.scatter_add_(0, src, expected_bonds)
        valency_penalty = F.relu(node_valency - 4.5).mean()
        
        # Combine losses
        loss = (1.0 * loss_nodes + 
                4.0 * loss_edges + 
                2.0 * valency_penalty + 
                1.0 * loss_conn + 
                0.5 * loss_dist +
                1.0 * loss_ring)

        
        self.log(f"{stage}_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log(f"{stage}_node_loss", loss_nodes, batch_size=batch_size)
        self.log(f"{stage}_edge_loss", loss_edges, batch_size=batch_size)
        self.log(f"{stage}_valency_penalty", valency_penalty, batch_size=batch_size)
        self.log(f"{stage}_ring_loss", loss_ring, batch_size=batch_size)
        self.log(f"{stage}_small_ring_penalty", small_ring_penalty, batch_size=batch_size)
        self.log(f"{stage}_macro_ring_penalty", macro_ring_penalty, batch_size=batch_size)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self.p_uncond)

    def validation_step(self, batch, batch_idx):
        # Validation shouldn't use CFG dropout to get an accurate conditional loss
        return self._shared_step(batch, "val", 0.0)
            
    def train(self, mode: bool = True):
        """Override train to force frozen judges to stay in eval mode."""
        super().train(mode)
        if hasattr(self, 'causal_judge') and self.causal_judge is not None:
            self.causal_judge.eval()
        if hasattr(self, 'property_judges'):
            self.property_judges.eval()  # Set the ModuleDict container to eval mode
            for judge in self.property_judges.values():
                judge.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-10
        )
        
        # Linear Warmup (10 epochs) then Cosine Annealing
        warmup_epochs = 15
        T_max = self.trainer.max_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            # Cosine decay
            progress = float(epoch - warmup_epochs) / float(max(1, T_max - warmup_epochs))
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
