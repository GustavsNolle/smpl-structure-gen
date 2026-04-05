"""Inference and Latent Space Analysis for Semi-Supervised Molecular Maps.

Calculates the KNN-Tanimoto Consistency metric to verify if the 5D latent space
respects fundamental chemical structural similarity.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from mol_prop_gnn.data.unified_dataset import (
    UNIFIED_DATASETS, 
    build_unified_dataframe, 
    preprocess_unified_dataset
)
from mol_prop_gnn.data.preprocessing import (
    get_node_feature_dim,
    get_edge_feature_dim
)
from mol_prop_gnn.models.joint_embedder import JointMolEmbedder
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule

# Backbone factory (mirrored from train_semi_supervised.py)
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.gat import MolGAT
from mol_prop_gnn.models.egnn import MolEGNN
from mol_prop_gnn.models.gine import MolGINE
from mol_prop_gnn.models.rgcn import MolRGCN

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_backbone(name: str, node_dim: int, edge_dim: int) -> nn.Module:
    """Factory to build the GNN encoder backbone."""
    hidden_dim = 256
    layers = 5

    if name == "gcn":
        return MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "gat":
        return MolGAT(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=64, heads=4, num_gnn_layers=layers)
    elif name == "egnn":
        return MolEGNN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=layers)
    elif name == "gine":
        return MolGINE(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "rgcn":
        return MolRGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=layers)
    else:
        raise ValueError(f"Unknown backbone model: {name}")


def get_fingerprints(smiles_list: list[str]) -> list[Any | None]:
    """Compute Morgan Fingerprints for a list of SMILES."""
    fps = []
    for s in tqdm(smiles_list, desc="Computing Fingerprints"):
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        else:
            fps.append(None)
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent Space Consistency Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "gat", "egnn", "rgcn", "gine"],
                        help="GNN backbone architecture used during training")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random molecules to sample")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for consistency metric")
    args = parser.parse_args()

    # 1. Prepare Data
    logger.info("Loading unified dataset...")
    df, _ = build_unified_dataframe(raw_dir="data/raw")
    
    if len(df) > args.samples:
        df_sample = df.sample(args.samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
    
    logger.info("Converting SMILES to Graphs...")
    graphs, _, _, _ = preprocess_unified_dataset(df_sample)
    smiles_list = [g.smiles for g in graphs]

    # 2. Reconstruct Model from Checkpoint
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    
    logger.info("Reconstructing model: %s", args.model)
    backbone = build_backbone(args.model, node_dim, edge_dim)
    joint_model = JointMolEmbedder(
        backbone=backbone,
        backbone_out_dim=backbone.out_channels,
        num_datasets=len(UNIFIED_DATASETS)
    )
    
    lit_module = JointSemiSupModule.load_from_checkpoint(
        args.checkpoint,
        model=joint_model
    )
    lit_module.eval()
    lit_module.freeze()

    # 3. Compute Latent Embeddings (5D bottleneck space)
    loader = DataLoader(graphs, batch_size=128, shuffle=False)
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference (Latent Space)"):
            emb = lit_module(batch)
            all_embeddings.append(emb)
    
    latent_space = torch.cat(all_embeddings, dim=0).cpu().numpy()

    # 4. Compute Structural Fingerprints
    fps = get_fingerprints(smiles_list)
    
    # Filter out any that failed conversion
    valid_mask = [fp is not None for fp in fps]
    fps = [fp for fp in fps if fp is not None]
    latent_space = latent_space[valid_mask]
    n_valid = len(fps)
    
    logger.info("Analyzing Consistency for %d molecules...", n_valid)

    # 5. Compute KNN-Tanimoto Consistency
    # Distance matrix in latent space
    latent_dist = squareform(pdist(latent_space, metric='euclidean'))
    
    consistencies = []
    for i in tqdm(range(n_valid), desc="Calculating Overlap"):
        # Structural neighbors (Tanimoto)
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        # argsort descending, skip index i
        struct_ranks = np.argsort(tanimoto_sims)[::-1]
        struct_list = [idx for idx in struct_ranks if idx != i][:args.k]
        struct_set = set(struct_list)
        
        # Latent neighbors (Euclidean distance)
        # argsort ascending, skip index i
        latent_ranks = np.argsort(latent_dist[i])
        latent_list = [idx for idx in latent_ranks if idx != i][:args.k]
        latent_set = set(latent_list)
        
        # Calculate intersection
        overlap = len(struct_set & latent_set)
        consistencies.append(overlap / args.k)

    avg_consistency = np.mean(consistencies)
    
    # 6. Correlation Analysis
    corr_matrix = pd.DataFrame(latent_space, columns=UNIFIED_DATASETS).corr()

    # 7. Final Results
    logger.info("=========================================")
    logger.info("    KNN-TANIMOTO CONSISTENCY REPORT      ")
    logger.info("=========================================")
    logger.info(" Backbone Model:  %s", args.model.upper())
    logger.info(" Checkpoint:      %s", Path(args.checkpoint).name)
    logger.info(" Sample Size:     %d", n_valid)
    logger.info(" K-Neighbors:     %d", args.k)
    logger.info("-----------------------------------------")
    logger.info(" AVG CONSISTENCY: %.4f", avg_consistency)
    logger.info(" (Random Chance:  %.4f)", args.k / n_valid)
    logger.info("=========================================")
    logger.info(" LATENT PROPERTY CORRELATIONS:")
    logger.info("\n" + str(corr_matrix.round(3)))
    logger.info("=========================================")


if __name__ == "__main__":
    main()
