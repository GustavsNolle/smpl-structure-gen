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
        
        # We will load the frozen causal judge dynamically in setup()
        # to ensure it's on the correct device.
        self.causal_judge = None
        self.causal_judge_model_id = causal_judge_model_id
        
        # Load ZINC marginals
        marginals_path = Path("data/zinc_marginals.pt")
                # 2. Hardcoded Dense ZINC Edge Marginals (Idealized Sparsity)
        # This bypasses the "Sparse Trap" where the model thinks bonds are common.
        # 95% No-Bond, 4% Single, 0.5% Double, 0.2% Triple, 0.2% Aromatic, 0.1% Other
        edge_m = torch.tensor([0.95, 0.04, 0.005, 0.002, 0.002, 0.001], device=self.device)
        self.register_buffer("edge_marginals", edge_m)
            
        # 3. Node Marginals
        node_m = torch.ones(self.num_node_classes, device=self.device) / self.num_node_classes
        self.register_buffer("node_marginals", node_m)
        
        # Load Node Marginals from file if exists, else fallback to uniform
        if marginals_path.exists():
            logger.info("Loading ZINC node marginals...")
            marginals = torch.load(marginals_path)
            self.register_buffer("node_marginals", marginals["node_marginals"])
        else:
            logger.warning("ZINC marginals not found! Falling back to uniform for nodes.")
            self.register_buffer("node_marginals", torch.ones(num_node_classes) / num_node_classes)

        # GPU Bottleneck Fix: Register noise schedule as buffer
        alpha_bar_all = get_noise_schedule(num_timesteps, torch.device('cpu'))
        self.register_buffer("alpha_bar_all", alpha_bar_all)

    def setup(self, stage=None):
        """Loads the pre-trained frozen Causal Judge from ClearML."""
        if self.causal_judge is None:
            logger.info(f"Loading frozen Causal Judge (ClearML ID: {self.causal_judge_model_id})...")
            
            # Use ClearML to fetch the local path of the downloaded model
            model_info = Model(model_id=self.causal_judge_model_id)
            local_path = model_info.get_local_copy()
            
            # Load the checkpoint
            ckpt = torch.load(local_path, map_location=self.device)
            
            # Retrieve the model config from hyperparameters
            model_config = ckpt.get("hyper_parameters", {}).get("model_config", {})
            
            # Extract values with fallbacks to defaults
            backbone_name = model_config.get("backbone_name", "gin")
            hidden_dim = model_config.get("hidden_dim", 256)
            num_layers = model_config.get("num_layers", 5)
            dropout = model_config.get("dropout", 0.3)
            
            # Auto-detect bottleneck_dim from state_dict if missing from model_config
            state_dict = ckpt["state_dict"]
            if "model.causal_head.weight" in state_dict:
                # Shape is [num_tasks, bottleneck_dim]
                bottleneck_dim = state_dict["model.causal_head.weight"].shape[1]
                logger.info(f"Auto-detected bottleneck_dim={bottleneck_dim} from state_dict.")
            else:
                bottleneck_dim = model_config.get("bottleneck_dim", self.hparams.causal_cond_dim)
            
            if bottleneck_dim != self.hparams.causal_cond_dim:
                logger.warning(
                    f"Causal Judge bottleneck ({bottleneck_dim}) does not match "
                    f"DiGress causal_cond_dim ({self.hparams.causal_cond_dim}). "
                    "This may cause a dimension mismatch in the generator forward pass!"
                )
                
            num_tasks = model_config.get("num_tasks", 21)
            deg = model_config.get("deg", None)
            
            judge_model = build_causal_model(
                backbone_name=backbone_name,
                node_dim=get_node_feature_dim(),
                edge_dim=get_edge_feature_dim(),
                num_tasks=num_tasks,
                bottleneck_dim=bottleneck_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                deg=deg
            )
            
            # Extract state dict for the model within the Lightning Module
            state_dict = ckpt["state_dict"]
            model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
            judge_model.load_state_dict(model_state_dict)
            
            self.causal_judge = judge_model
            self.causal_judge.to(self.device)
            self.causal_judge.eval()
            
            # Freeze the judge
            for param in self.causal_judge.parameters():
                param.requires_grad = False
                
            logger.info("Causal Judge loaded and frozen successfully.")

    def forward(self, batch):
        # We don't typically use the forward method directly in diffusion modules.
        pass

    def get_causal_embedding(self, batch) -> torch.Tensor:
        """Extracts the 128/256-dim causal embedding from the frozen judge."""
        from torch_geometric.nn import global_mean_pool
        
        self.causal_judge.eval()
        with torch.no_grad():
            # 1. Encode graph structure via backbone
            h_node = self.causal_judge.backbone.encode(
                x=batch.x, 
                edge_index=batch.edge_index, 
                edge_attr=batch.edge_attr, 
                batch=batch.batch
            )
            
            # 2. Predict node mask
            mask_logits = self.causal_judge.extractor(h_node)
            mask = torch.sigmoid(mask_logits)
            
            # 3. Apply mask to get Causal Subgraph features
            h_node_c = h_node * mask
            
            # 4. Pool to graph level
            h_graph_c = global_mean_pool(h_node_c, batch.batch)
            
            # 5. Project through bottleneck to get final causal_emb
            causal_emb = self.causal_judge.causal_bottleneck(h_graph_c)
            
        return causal_emb

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
            
        # 3b. Denseify graphs for discrete diffusion (fully connected topology)
        edge_index_dense, E_true = denseify_graphs(
            batch_edge_index=batch.edge_index,
            batch_edge_attr=batch.edge_attr,
            batch_idx=batch.batch,
            num_edge_classes=self.num_edge_classes
        )
            
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
        # L = D - A
        D_dense = torch.diag_embed(adj_dense.sum(dim=-1))
        L_dense = D_dense - adj_dense
        # Fiedler values: second smallest eigenvalues
        evals_dense = torch.linalg.eigvalsh(L_dense)
        if evals_dense.shape[1] > 1:
            lambda2 = evals_dense[:, 1]
            loss_conn = F.relu(0.02 - lambda2).mean() # Encourage connectivity
        else:
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

        # D. Valency Penalty
        src, dst = edge_index_dense
        node_valency = torch.zeros(X_noisy.shape[0], device=self.device)
        node_valency.scatter_add_(0, src, expected_bonds)
        valency_penalty = F.relu(node_valency - 4.5).mean()
        
        # Combine losses: Shock Therapy to break conditioning deafness
        loss = (1.0 * loss_nodes + 
                4.0 * loss_edges + 
                2.0 * valency_penalty + 
                1.0 * loss_conn + 
                0.5 * loss_dist)
        
        self.log(f"{stage}_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log(f"{stage}_node_loss", loss_nodes, batch_size=batch_size)
        self.log(f"{stage}_edge_loss", loss_edges, batch_size=batch_size)
        self.log(f"{stage}_valency_penalty", valency_penalty, batch_size=batch_size)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self.p_uncond)

    def validation_step(self, batch, batch_idx):
        # Validation shouldn't use CFG dropout to get an accurate conditional loss
        return self._shared_step(batch, "val", 0.0)

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
