#!/usr/bin/env python3
"""Inference script for Anchor-Guided Molecule Generation.

Uses the trained Multi-Judge DiGress model to generate molecules conditionally
guided by 20 specific anchor molecules (10 basics, 10 ClinTox).
It saves all valid molecules to a CSV and generates 4 image grids.

Usage:
  python scripts/inference_guided.py --num_samples 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig, QED as QED_module
from rdkit.Chem import Draw

# RDKit SA Score calculator
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from mol_prop_gnn.training.digress_module import DiGressModule
from mol_prop_gnn.training.digress_sampler import sample_digress_graphs
from mol_prop_gnn.utils.molecule_decoder import graph_to_smiles
from mol_prop_gnn.data.preprocessing import smiles_to_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


ANCHORS = {
    "Gemini_hal_1": "Nc1ncnc2[nH]cc(C34CC(C3)(C4)c5ccc(F)cc5)c12",
    "Gemini_hal_2": "O=C(C12CC3CC(C1)CC(C2)C3)N4CC5CC5C4",
    "JNJ": "O=C(C1=CNC2=C1C=C(OC(F)(F)F)C=C2)C(C3=CC=C(Cl)C=C3OC)NC4=CC(S(=O)(C)=O)=CC(OC)=C4",
    "BASE": "O=S(C(C=C1)=CC=C1OCC)(N2N=C(C=C2N)OC(C3=CC=C(C=C3)C4=CC=CC=C4)=O)=O",
    "Edited": "Nc1ncnc2[nH]cc(C34CC(C3)(C4)c5ccc(O)cc5)c12",
    "Ed_2": "Nc1ncnc2[nH]cc(C34CC(C3)(C4)c5cc(F)c(F)cc5)c12",
    "Ed_3": "Nc1ncnc2[nH]c(O)c(C34CC(C3)(C4)c5ccc(F)cc5)c12",
    "ED_2_1": "Nc1ncnc2[nH]c(O)c(C34CC(C3)(C4)c5cc(F)c(F)cc5)c12",
    "ED_2_2": "NC(=O)c1c(C23CC(C2)(C3)c4cc(F)c(F)cc4)c[nH]c2ncnc(N)c12",
    "ED_2_3": "Nc1ncnc2[nH]cc(C34CC(C3)(C4)c5cc(C(F)(F)F)ccc5)c12",
    "ED_2_4": "Nc1ncnc2[nH]cc(C34CC(C3)(C4)c5cc(F)c(F)cn5)c12",
    # "Sample_generated": "CCN(C)C(=O)CC1C2CCCCC2NC2CCC21",
    # "Gemini_prompted": "COc1cc(S(C)(=O)=N)cc(N[C@@H](c2cc(Cl)cnc2OC)C(=O)c3c[nH]c4ccc(S(F)(F)(F)(F)F)cc34)c1",
    # "Sample_gen_highsas": "CCC1(C)CC23CC4C5(CC56CCC(CN)C6)C2C413"
}

def compute_properties(smiles: str) -> dict | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "smiles": smiles,
            "SAS": sascorer.calculateScore(mol),
            "qed": QED_module.qed(mol),
            "logP": Descriptors.MolLogP(mol),
        }
    except Exception:
        return None

def load_generator(device: torch.device, clearml_id: str = None) -> DiGressModule:
    if clearml_id:
        from clearml import Model
        logger.info(f"Fetching DiGress checkpoint from ClearML (ID: {clearml_id})...")
        model_info = Model(model_id=clearml_id)
        latest_last_ckpt = model_info.get_local_copy()
    else:
        ckpt_dir = Path("checkpoints/phase3_digress")
        latest_last_ckpt = None
        if ckpt_dir.exists():
            last_ckpts = list(ckpt_dir.glob("last*.ckpt"))
            if last_ckpts:
                latest_last_ckpt = max(last_ckpts, key=lambda p: p.stat().st_mtime)
                
        if not latest_last_ckpt:
            raise FileNotFoundError("Could not find a valid checkpoint in checkpoints/phase3_digress!")
        
    logger.info(f"Loading DiGress from {latest_last_ckpt}...")
    ckpt = torch.load(latest_last_ckpt, map_location=device, weights_only=False)
    
    hparams = ckpt.get("hyper_parameters", {})
    lit_module = DiGressModule(
        num_node_classes=hparams.get("num_node_classes", 11),
        num_edge_classes=hparams.get("num_edge_classes", 6),
        hidden_dim=hparams.get("hidden_dim", 128),
        causal_cond_dim=hparams.get("causal_cond_dim", 448),
        num_layers=hparams.get("num_layers", 5),
        num_heads=hparams.get("num_heads", 4),
        num_timesteps=hparams.get("num_timesteps", 1000),
        causal_judge_model_id=hparams.get("causal_judge_model_id", "94f148c657ed4b7b8fdaa54b4ad2bdd3"),
        property_judge_ids=hparams.get("property_judge_ids", {})
    )
    
    lit_module.load_state_dict(ckpt["state_dict"], strict=False)
    lit_module = lit_module.to(device)
    lit_module.eval()
    
    # Must run setup to load the frozen judges
    lit_module.setup()
    
    return lit_module

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500, help="Number of generations per anchor")
    parser.add_argument("--batch_size", type=int, default=50, help="Generation batch size")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG guidance scale")
    parser.add_argument("--clearml_id", type=str, default=None, help="ClearML Model ID to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create outputs directory
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "gemini_valid_molecules.csv"
    filtered_csv_path = out_dir / "gemini_filtered_molecules.csv"

    model = load_generator(device, args.clearml_id)

    all_valid_df = []
    all_filtered_df = []
    
    # 1 image with 4 anchors
    anchor_items = list(ANCHORS.items())
    
    for img_idx in range(1):
        batch_anchors = anchor_items
        
        grid_mols = []
        grid_legends = []
        
        for name, smi in batch_anchors:
            logger.info(f"Processing anchor: {name} ({smi})")
            
            # Draw anchor first
            anchor_mol = Chem.MolFromSmiles(smi)
            anchor_props = compute_properties(smi)
            s_a = anchor_props["SAS"] if anchor_props else 0
            q_a = anchor_props["qed"] if anchor_props else 0
            l_a = anchor_props["logP"] if anchor_props else 0
            
            if anchor_props:
                ap_copy = anchor_props.copy()
                ap_copy["anchor"] = f"{name}_BASE_REFERENCE"
                all_filtered_df.append(ap_copy)
            
            grid_mols.append(anchor_mol)
            grid_legends.append(f"ANCHOR: {name}\nSA:{s_a:.1f} QED:{q_a:.2f} L:{l_a:.1f}")
            
            # Extract condition
            data = smiles_to_graph(smi)
            if data is None:
                logger.error(f"Failed to featurize anchor {name}")
                continue
            
            data = data.to(device)
            # Create a dummy batch vector
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            
            with torch.no_grad():
                c_anchor = model.get_causal_embedding(data)
                
            # Sample molecules
            valid_mols = []
            
            batches = args.num_samples // args.batch_size
            rem = args.num_samples % args.batch_size
            batch_sizes = [args.batch_size] * batches + ([rem] if rem > 0 else [])
            
            for bs in tqdm(batch_sizes, desc=f"Generating for {name}"):
                c_batch = c_anchor.expand(bs, -1)
                # Assume mean node count of ~25 based on dataset
                node_counts = torch.randint(15, 35, (bs,), device=device)
                
                with torch.no_grad():
                    X_gen, E_gen, edge_index_gen, batch_idx_gen = sample_digress_graphs(
                        model=model.model,
                        c=c_batch,
                        num_nodes_per_graph=node_counts.tolist(),
                        node_marginals=model.node_marginals,
                        edge_marginals=model.edge_marginals,
                        num_node_classes=model.num_node_classes,
                        num_edge_classes=model.num_edge_classes,
                        num_timesteps=model.num_timesteps,
                        guidance_scale=args.guidance_scale
                    )
                
                for i in range(bs):
                    mask = (batch_idx_gen == i)
                    n_idx = torch.nonzero(mask).squeeze(-1)
                    if len(n_idx) == 0: continue
                    
                    nt = X_gen[mask].reshape(-1)
                    edge_mask = (batch_idx_gen[edge_index_gen[0]] == i)
                    ei = edge_index_gen[:, edge_mask]
                    ei = ei - n_idx.min().item()
                    et = E_gen[edge_mask].reshape(-1)
                    
                    smiles = graph_to_smiles(nt, ei, et)
                    if smiles:
                        props = compute_properties(smiles)
                        if props:
                            props["anchor"] = name
                            valid_mols.append(props)
            
            if valid_mols:
                all_valid_df.extend(valid_mols)
            
            # Filter and pick best 8
            # SAscore under 4.5, qed score over 0.55
            filtered = [m for m in valid_mols if m["SAS"] < 4.5 and m["qed"] > 0.55 and m["logP"] < 5 and m["logP"] > 0]
            logger.info(f"Anchor {name}: {len(valid_mols)} valid total, {len(filtered)} passed filters.")
            
            if filtered:
                all_filtered_df.extend(filtered)
            
            # Sort by QED descending
            filtered = sorted(filtered, key=lambda x: x["qed"], reverse=True)
            
            best_8 = filtered[:8]
            
            for m in best_8:
                mol = Chem.MolFromSmiles(m["smiles"])
                grid_mols.append(mol)
                grid_legends.append(f"SA:{m['SAS']:.1f} QED:{m['qed']:.2f} L:{m['logP']:.1f}")
                
            # Pad if fewer than 8
            for _ in range(8 - len(best_8)):
                grid_mols.append(Chem.MolFromSmiles("C"))
                grid_legends.append("N/A")
                
        # Draw image for all 4 anchors
        logger.info(f"Drawing final grid image...")
        img = Draw.MolsToGridImage(
            grid_mols,
            molsPerRow=9, # 1 anchor + 8 molecules
            subImgSize=(300, 300),
            legends=grid_legends
        )
        img_path = out_dir / f"grid_gemini_anchors.png"
        img.save(img_path)
        logger.info(f"Saved grid to {img_path}")
        
    # Save CSVs
    if all_valid_df:
        df = pd.DataFrame(all_valid_df)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} total valid generated molecules to {csv_path}")
        
    if all_filtered_df:
        df_filtered = pd.DataFrame(all_filtered_df)
        df_filtered.to_csv(filtered_csv_path, index=False)
        logger.info(f"Saved {len(df_filtered)} filtered molecules to {filtered_csv_path}")

if __name__ == "__main__":
    main()
