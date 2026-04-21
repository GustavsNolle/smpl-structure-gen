import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mol_prop_gnn.data.rdkit_zinc import RDKitZINC

def calculate_marginals():
    root = "data/ZINC"
    dataset = RDKitZINC(root)
    
    num_node_classes = 11
    num_edge_classes = 6
    
    node_counts = torch.zeros(num_node_classes)
    edge_counts = torch.zeros(num_edge_classes)
    
    print(f"Calculating marginals for {len(dataset)} molecules...")
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        # Nodes (categorical indices)
        x_indices = torch.argmax(data.x[:, :10], dim=1)
        # We assume anything not in the first 10 is 'other/empty' (class 10)
        # Wait, let's be more precise. If all 10 are 0, it's class 10.
        node_types = torch.where(data.x[:, :10].sum(dim=1) > 0, x_indices, torch.tensor(10))
        node_counts.scatter_add_(0, node_types, torch.ones_like(node_types, dtype=torch.float))
        
        # Edges (categorical indices)
        # In the sparse dataset, all edges are valid bonds (0-3) or other (4)
        # Class 5 is 'no bond', which is NOT in the sparse dataset.
        # However, to get the TRUE marginal for a dense model, we must count ALL pairs.
        
        num_nodes = data.x.shape[0]
        num_possible_edges = num_nodes * (num_nodes - 1)
        num_existing_edges = data.edge_index.shape[1]
        
        # Existing bonds
        # Shifted by 1 so 0 is reserved for 'No Bond'
        e_types = torch.argmax(data.edge_attr[:, :5], dim=1) + 1
        edge_counts.scatter_add_(0, e_types, torch.ones_like(e_types, dtype=torch.float))
        
        # 'No bond' (class 0)
        num_no_bonds = num_possible_edges - num_existing_edges
        edge_counts[0] += num_no_bonds
        
    node_marginals = node_counts / node_counts.sum()
    edge_marginals = edge_counts / edge_counts.sum()
    
    print("\nNode Marginals (0-10):")
    print(node_marginals.tolist())
    
    print("\nEdge Marginals (0-5):")
    print(edge_marginals.tolist())
    
    # Save to a file for use in training
    torch.save({
        'node_marginals': node_marginals,
        'edge_marginals': edge_marginals
    }, "data/zinc_marginals.pt")
    print("\nSaved marginals to data/zinc_marginals.pt")

if __name__ == "__main__":
    calculate_marginals()
