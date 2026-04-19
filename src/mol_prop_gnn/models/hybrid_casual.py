import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class CausalContrastiveUncertaintyEmbedder(nn.Module):
    """Hybrid novel method: Causal mask + Contrastive learning + Uncertainty weighting.
    Combines three existing techniques into one novel architecture:
    1. Causal subgraph mask 
    2. Contrastive learning between causal/environment views 
    3. Uncertainty-weighted multi-task loss 
    """
    
    def __init__(self, backbone, backbone_out_dim, num_datasets=18, bottleneck_dim=256, dropout=0.3, contrastive_temp=0.07):
        super().__init__()
        self.backbone = backbone
        self.num_datasets = num_datasets
        self.contrastive_temp = contrastive_temp
        
        # Causal mask extractor
        self.extractor = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.ReLU(),
            nn.Linear(backbone_out_dim // 2, 1)
        )
        
        # Shared bottlenecks
        self.causal_bottleneck = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, bottleneck_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.env_bottleneck = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, bottleneck_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        
        # Task heads
        self.causal_head = nn.Linear(bottleneck_dim, num_datasets)
        self.env_head = nn.Linear(bottleneck_dim, num_datasets)
        
        # Uncertainty log variances (learnable per task)
        self.log_vars = nn.Parameter(torch.zeros(num_datasets))
        
    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        # Encode
        h_node = self.backbone.encode(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, **kwargs)
        
        # Causal mask
        mask_logits = self.extractor(h_node)
        mask = torch.sigmoid(mask_logits)
        
        # Split into causal and environment subgraphs
        h_node_c = h_node * mask
        h_node_e = h_node * (1 - mask)
        
        # Pool
        h_graph_c = global_mean_pool(h_node_c, batch)
        h_graph_e = global_mean_pool(h_node_e, batch)
        
        # Bottleneck
        h_shared_c = self.causal_bottleneck(h_graph_c)
        h_shared_e = self.env_bottleneck(h_graph_e)
        
        # Contrastive loss between causal and environment views 
        # Normalize for cosine similarity
        z_c = F.normalize(h_shared_c, dim=1)
        z_e = F.normalize(h_shared_e, dim=1)
        contrastive_loss = -torch.log(torch.exp((z_c * z_e).sum(dim=1) / self.contrastive_temp).mean())
        
        # Predictions
        pred_c = self.causal_head(h_shared_c)
        pred_e = self.env_head(h_shared_e)
        
        return pred_c, pred_e, mask, contrastive_loss, self.log_vars
    
    @property
    def out_channels(self):
        return self.num_datasets
