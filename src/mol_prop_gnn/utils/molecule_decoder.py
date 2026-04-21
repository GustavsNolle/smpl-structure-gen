import torch
from rdkit import Chem

# Matches the preprocessing allowlists in src/mol_prop_gnn/data/preprocessing.py
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [
    None, # 0: No Bond
    Chem.BondType.SINGLE, 
    Chem.BondType.DOUBLE, 
    Chem.BondType.TRIPLE, 
    Chem.BondType.AROMATIC
]

def graph_to_mol(
    node_types: torch.Tensor, 
    edge_index: torch.Tensor, 
    edge_types: torch.Tensor,
) -> Chem.RWMol:
    """Converts predicted categorical nodes and edges back to an RDKit Mol object."""
    mol = Chem.RWMol()
    
    # 1. Add atoms
    idx_map = {}
    for i, nt in enumerate(node_types.tolist()):
        if nt < len(ATOM_LIST):
            atom = Chem.Atom(ATOM_LIST[nt])
            idx = mol.AddAtom(atom)
            idx_map[i] = idx
            
    # 2. Add edges
    added_bonds = set()
    for e in range(edge_index.shape[1]):
        u, v = edge_index[0, e].item(), edge_index[1, e].item()
        et = edge_types[e].item()
        
        # Undirected graph check to avoid adding duplicate bonds
        if u < v and u in idx_map and v in idx_map and 0 < et < len(BOND_TYPES):
            u_rdkit = idx_map[u]
            v_rdkit = idx_map[v]
            if (u_rdkit, v_rdkit) not in added_bonds:
                try:
                    mol.AddBond(u_rdkit, v_rdkit, BOND_TYPES[et])
                    added_bonds.add((u_rdkit, v_rdkit))
                    
                    # Force atoms to be aromatic if the bond is aromatic
                    if BOND_TYPES[et] == Chem.BondType.AROMATIC:
                        mol.GetAtomWithIdx(u_rdkit).SetIsAromatic(True)
                        mol.GetAtomWithIdx(v_rdkit).SetIsAromatic(True)
                except Exception:
                    pass
    return mol

def graph_to_smiles(
    node_types: torch.Tensor, 
    edge_index: torch.Tensor, 
    edge_types: torch.Tensor,
) -> str | None:
    """Converts predicted categorical nodes and edges back to a SMILES string."""
    mol = graph_to_mol(node_types, edge_index, edge_types)
                    
    # 3. Sanitize and Convert to SMILES
    try:
        mol = mol.GetMol()
        # Sanitize ensures valence is physically possible
        Chem.SanitizeMol(mol)
        
        # Extract only the largest connected fragment to avoid "floating debris"
        from rdkit.Chem import rdmolops
        frags = rdmolops.GetMolFrags(mol, asMols=True)
        if len(frags) == 0:
            return None
            
        # Find the fragment with the most heavy atoms
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        
        smiles = Chem.MolToSmiles(largest_frag)
        if len(smiles) > 0:
            return smiles
        return None
    except Exception:
        return None
