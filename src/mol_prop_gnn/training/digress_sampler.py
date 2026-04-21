import torch
import torch.nn.functional as F
from mol_prop_gnn.training.digress_module import get_noise_schedule

@torch.no_grad()
def sample_digress_graphs(
    model, 
    c: torch.Tensor, 
    num_nodes_per_graph: list[int], 
    node_marginals: torch.Tensor,
    edge_marginals: torch.Tensor,
    num_node_classes: int = 11, 
    num_edge_classes: int = 6, 
    num_timesteps: int = 1000, 
    guidance_scale: float = 3.0
):
    """Generates graphs from noise biased towards marginals using the reverse process."""
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
    # This ensures p(u, v) == p(v, u) and E(u, v) == E(v, u)
    row, col = edge_index
    num_total_nodes = batch_idx.shape[0]
    edge_ids = row * num_total_nodes + col
    rev_edge_ids = col * num_total_nodes + row
    
    # perm maps each edge to its reverse: edge_index[:, perm] == edge_index[[1, 0], :]
    sorted_edge_ids, sorted_indices = torch.sort(edge_ids)
    perm = sorted_indices[torch.searchsorted(sorted_edge_ids, rev_edge_ids)]
    
    # 2. Initialize X_T and E_T from Marginal Distribution
    X = torch.multinomial(node_marginals.expand(batch_idx.shape[0], -1), num_samples=1).squeeze(-1)
    E = torch.multinomial(edge_marginals.expand(edge_index.shape[1], -1), num_samples=1).squeeze(-1)
    
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
        
        # The "Hockey Stick" Temperature Schedule
        if t_step > 20:
            # Full creativity during the scaffold-building phase
            temp = 1.0
        else:
            # Exponentially freeze at the end to lock in structural choices
            temp = 0.05 + 0.95 * ((t_step / 20.0) ** 2)

        p_X0 = F.softmax(node_logits / temp, dim=-1)
        p_E0 = F.softmax(edge_logits / temp, dim=-1)
        
        # Symmetrize edge probabilities
        p_E0 = 0.5 * (p_E0 + p_E0[perm])
        
        if t_step == 1:
            p_Xt_minus_1_X = p_X0
            p_Et_minus_1_E = p_E0
        else:
            # 1. Get alpha_t and alpha_bar_t-1
            # alpha_bar_all: [T]
            # t_step is 1-indexed. t_step=2 means we are at t=2 and want to go to t=1.
            # alpha_bar_t is alpha_bar_all[t_step-1]
            # alpha_bar_tm1 is alpha_bar_all[t_step-2]
            
            # For linear schedule: alpha_t = alpha_bar_t / alpha_bar_t-1
            alpha_bar_t = alpha_bar_all[t_step - 1]
            alpha_bar_tm1 = alpha_bar_all[t_step - 2]
            alpha_t = alpha_bar_t / alpha_bar_tm1
            
            # 2. Compute posterior for nodes
            # q(X_{t-1} | X_t, X_0) \propto Q_t(X_t | X_{t-1}) * Q_bar_tm1(X_{t-1} | X_0)
            
            # Q_t(i | j) = alpha_t * (i == j) + (1 - alpha_t) * m_i
            m_n = node_marginals # [num_classes]
            num_n = m_n.shape[0]
            
            # qt_xt_j: [N, num_node_classes]
            # Probabilities of X_t given potential states j in {0...K-1}
            # qt_xt_j[n, j] = alpha_t if X[n] == j else 0, plus (1 - alpha_t) * m_{X[n]}
            # Wait, the target state is X_t, the source is X_{t-1}=j.
            # So Q_t(X_t | j) = alpha_t * (X_t == j) + (1 - alpha_t) * m_{X_t}
            
            # Construction:
            # All entries get (1 - alpha_t) * m[X_t]
            qt_xt_j = (1.0 - alpha_t) * m_n[X].view(-1, 1).expand(-1, num_n)
            # Add alpha_t to the diagonal where j == X_t
            qt_xt_j.scatter_add_(1, X.view(-1, 1), torch.full((X.shape[0], 1), alpha_t.item(), device=device))
            
            # Q_bar_tm1_X0: [N, K]
            # P(X_{t-1} | X_0) = alpha_bar_tm1 * p_X0 + (1 - alpha_bar_tm1) * m_n
            qbar_tm1_x0 = alpha_bar_tm1 * p_X0 + (1.0 - alpha_bar_tm1) * m_n.view(1, -1)
            
            # Posterior = Normalization(qt_xt_j * qbar_tm1_x0)
            p_Xt_minus_1_X = qt_xt_j * qbar_tm1_x0
            p_Xt_minus_1_X = p_Xt_minus_1_X / (p_Xt_minus_1_X.sum(dim=-1, keepdim=True) + 1e-9)
            
            # 3. Compute posterior for edges
            m_e = edge_marginals
            num_e = m_e.shape[0]
            
            qt_et_j = (1.0 - alpha_t) * m_e[E].view(-1, 1).expand(-1, num_e)
            qt_et_j.scatter_add_(1, E.view(-1, 1), torch.full((E.shape[0], 1), alpha_t.item(), device=device))
            
            qbar_tm1_e0 = alpha_bar_tm1 * p_E0 + (1.0 - alpha_bar_tm1) * m_e.view(1, -1)
            
            p_Et_minus_1_E = qt_et_j * qbar_tm1_e0
            p_Et_minus_1_E = p_Et_minus_1_E / (p_Et_minus_1_E.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Symmetrize edge posterior
            p_Et_minus_1_E = 0.5 * (p_Et_minus_1_E + p_Et_minus_1_E[perm])
            
        # Sample next state
        if t_step < 10:
            # Argmax trick for final steps to ensure valency/structural stability
            X = torch.argmax(p_Xt_minus_1_X, dim=-1)
            E = torch.argmax(p_Et_minus_1_E, dim=-1)
        else:
            X = torch.multinomial(p_Xt_minus_1_X, num_samples=1).squeeze(-1)
            E = torch.multinomial(p_Et_minus_1_E, num_samples=1).squeeze(-1)
            
        # Force E to be symmetric: E(u, v) = E(v, u)
        # We take the upper triangle value and copy it to the lower triangle
        E = torch.where(row < col, E, E[perm])
        
    return X, E, edge_index, batch_idx
