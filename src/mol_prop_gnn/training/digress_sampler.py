import torch
import torch.nn.functional as F
from mol_prop_gnn.training.digress_module import get_noise_schedule

import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def sample_digress_graphs(
    model, 
    c: torch.Tensor, 
    num_nodes_per_graph: list[int], 
    node_marginals: torch.Tensor,
    edge_marginals: torch.Tensor,
    num_node_classes: int = 11, 
    num_edge_classes: int = 6, 
    num_timesteps: int = 100, 
    guidance_scale: float = 3.0
):
    """Generates graphs from noise biased towards marginals using the reverse process.
    
    Includes numerical robustness guards to prevent CUDA crashes from
    NaN logits or out-of-range indices.
    """
    device = c.device
    batch_size = c.shape[0]
    
    # 1. Create fully connected graphs
    batch_idx_list = []
    edge_index_list = []
    offset = 0
    for i, n_nodes in enumerate(num_nodes_per_graph):
        batch_idx_list.append(torch.full((n_nodes,), i, dtype=torch.long, device=device))
        idx = torch.arange(n_nodes, device=device)
        u = idx.repeat(n_nodes)
        v = idx.repeat_interleave(n_nodes)
        mask = u != v
        edge_index_list.append(torch.stack([u[mask] + offset, v[mask] + offset], dim=0))
        offset += n_nodes
        
    batch_idx = torch.cat(batch_idx_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    
    # 1.5. Find symmetric edge indices for symmetrization
    row, col = edge_index
    num_total_nodes = batch_idx.shape[0]
    edge_ids = row * num_total_nodes + col
    rev_edge_ids = col * num_total_nodes + row
    
    sorted_edge_ids, sorted_indices = torch.sort(edge_ids)
    perm = sorted_indices[torch.searchsorted(sorted_edge_ids, rev_edge_ids)]
    
    # 2. Initialize X_T and E_T from Marginal Distribution
    node_probs = torch.clamp(node_marginals, min=1e-6)
    node_probs = node_probs / node_probs.sum()
    edge_probs = torch.clamp(edge_marginals, min=1e-6)
    edge_probs = edge_probs / edge_probs.sum()
    
    X = torch.multinomial(node_probs.expand(batch_idx.shape[0], -1), num_samples=1).squeeze(-1)
    E = torch.multinomial(edge_probs.expand(edge_index.shape[1], -1), num_samples=1).squeeze(-1)
    
    # Clamp to valid ranges
    X = torch.clamp(X, 0, num_node_classes - 1)
    E = torch.clamp(E, 0, num_edge_classes - 1)
    
    # Symmetrize initial noise E: E(u, v) = E(v, u)
    E = torch.where(row < col, E, E[perm])
    
    alpha_bar_all = get_noise_schedule(num_timesteps, device)
    
    # 3. Reverse Diffusion Loop
    for t_step in reversed(range(1, num_timesteps + 1)):
        t = torch.full((batch_size,), t_step - 1, dtype=torch.long, device=device)
        
        # Predict logits for X_0 and E_0 using CFG
        node_logits, edge_logits = model.predict_cfg_logits(
            X=X, E=E, edge_index=edge_index, batch_idx=batch_idx, t=t, c=c, guidance_scale=guidance_scale
        )
        
        # CRITICAL: Clamp logits to prevent NaN from high CFG scale.
        # With guidance_scale=7.0, logits can reach ±1000+ causing softmax overflow.
        node_logits = torch.clamp(node_logits, -30.0, 30.0)
        edge_logits = torch.clamp(edge_logits, -30.0, 30.0)
        
        # Replace any NaN with zeros (fallback to uniform after softmax)
        node_logits = torch.nan_to_num(node_logits, nan=0.0)
        edge_logits = torch.nan_to_num(edge_logits, nan=0.0)
        
        # The "Hockey Stick" Temperature Schedule
        if t_step > 20:
            temp = 1.0
        else:
            temp = 0.05 + 0.95 * ((t_step / 20.0) ** 2)

        p_X0 = F.softmax(node_logits / temp, dim=-1)
        p_E0 = F.softmax(edge_logits / temp, dim=-1)
        
        # Symmetrize edge probabilities
        p_E0 = 0.5 * (p_E0 + p_E0[perm])
        
        if t_step == 1:
            p_Xt_minus_1_X = p_X0
            p_Et_minus_1_E = p_E0
        else:
            alpha_bar_t = alpha_bar_all[t_step - 1]
            alpha_bar_tm1 = alpha_bar_all[t_step - 2]
            alpha_t = alpha_bar_t / alpha_bar_tm1
            
            # Posterior for nodes
            m_n = node_probs
            num_n = m_n.shape[0]
            
            qt_xt_j = (1.0 - alpha_t) * m_n[X].view(-1, 1).expand(-1, num_n)
            qt_xt_j.scatter_add_(1, X.view(-1, 1), torch.full((X.shape[0], 1), alpha_t.item(), device=device))
            
            qbar_tm1_x0 = alpha_bar_tm1 * p_X0 + (1.0 - alpha_bar_tm1) * m_n.view(1, -1)
            
            p_Xt_minus_1_X = qt_xt_j * qbar_tm1_x0
            p_Xt_minus_1_X = p_Xt_minus_1_X / (p_Xt_minus_1_X.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Posterior for edges
            m_e = edge_probs
            num_e = m_e.shape[0]
            
            qt_et_j = (1.0 - alpha_t) * m_e[E].view(-1, 1).expand(-1, num_e)
            qt_et_j.scatter_add_(1, E.view(-1, 1), torch.full((E.shape[0], 1), alpha_t.item(), device=device))
            
            qbar_tm1_e0 = alpha_bar_tm1 * p_E0 + (1.0 - alpha_bar_tm1) * m_e.view(1, -1)
            
            p_Et_minus_1_E = qt_et_j * qbar_tm1_e0
            p_Et_minus_1_E = p_Et_minus_1_E / (p_Et_minus_1_E.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Symmetrize edge posterior
            p_Et_minus_1_E = 0.5 * (p_Et_minus_1_E + p_Et_minus_1_E[perm])
        
        # Ensure probabilities are valid before sampling
        p_Xt_minus_1_X = torch.clamp(p_Xt_minus_1_X, min=1e-10)
        p_Xt_minus_1_X = p_Xt_minus_1_X / p_Xt_minus_1_X.sum(dim=-1, keepdim=True)
        p_Et_minus_1_E = torch.clamp(p_Et_minus_1_E, min=1e-10)
        p_Et_minus_1_E = p_Et_minus_1_E / p_Et_minus_1_E.sum(dim=-1, keepdim=True)
            
        # Sample next state
        if t_step < 10:
            X = torch.argmax(p_Xt_minus_1_X, dim=-1)
            E = torch.argmax(p_Et_minus_1_E, dim=-1)
        else:
            X = torch.multinomial(p_Xt_minus_1_X, num_samples=1).squeeze(-1)
            E = torch.multinomial(p_Et_minus_1_E, num_samples=1).squeeze(-1)
        
        # CRITICAL: Clamp indices after every sampling step to prevent
        # out-of-bounds access in nn.Embedding on the next iteration
        X = torch.clamp(X, 0, num_node_classes - 1)
        E = torch.clamp(E, 0, num_edge_classes - 1)
            
        # Force E to be symmetric
        E = torch.where(row < col, E, E[perm])
        
    return X, E, edge_index, batch_idx
