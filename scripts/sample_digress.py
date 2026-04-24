#!/usr/bin/env python3
"""Sample valid molecules from a trained DiGress generator.

Loads a DiGress checkpoint from ClearML, generates molecules in batches
until a target number of valid SMILES is reached, computes SA/QED/LogP
for each, and saves the results as a CSV for mini judge training.

Usage:
  python scripts/sample_digress.py --clearml_id 152d4e4b1f4a42608f91d98d4fdf1b6f --target 20000
  python scripts/sample_digress.py --clearml_id 152d4e4b1f4a42608f91d98d4fdf1b6f --target 20000 --guidance_scale 0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig, QED as QED_module

# RDKit SA Score calculator
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from clearml import Model

from mol_prop_gnn.training.digress_module import DiGressModule
from mol_prop_gnn.training.digress_sampler import sample_digress_graphs
from mol_prop_gnn.utils.molecule_decoder import graph_to_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def compute_properties(smiles: str) -> dict | None:
    """Compute SA, QED, and LogP from a SMILES string."""
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


def load_generator(clearml_id: str, device: torch.device) -> DiGressModule:
    """Load DiGress generator from ClearML checkpoint."""
    logger.info(f"Fetching DiGress checkpoint from ClearML (ID: {clearml_id})...")
    model_info = Model(model_id=clearml_id)
    local_path = model_info.get_local_copy()
    logger.info(f"Checkpoint downloaded to: {local_path}")

    # Load checkpoint manually with strict=False because the checkpoint 
    # contains causal_judge weights that are loaded dynamically in setup().
    # We only need the generator (self.model) for sampling.
    ckpt = torch.load(local_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    
    lit_module = DiGressModule(
        num_node_classes=hparams.get("num_node_classes", 11),
        num_edge_classes=hparams.get("num_edge_classes", 6),
        hidden_dim=hparams.get("hidden_dim", 256),
        causal_cond_dim=hparams.get("causal_cond_dim", 256),
        num_layers=hparams.get("num_layers", 6),
        num_heads=hparams.get("num_heads", 4),
        num_timesteps=hparams.get("num_timesteps", 100),
        learning_rate=hparams.get("learning_rate", 1e-5),
        causal_judge_model_id=hparams.get("causal_judge_model_id", ""),
        p_uncond=hparams.get("p_uncond", 0.25),
    )
    
    # Load state dict, skipping causal_judge keys (not needed for sampling)
    state_dict = ckpt["state_dict"]
    missing, unexpected = lit_module.load_state_dict(state_dict, strict=False)
    
    # Filter out expected missing/unexpected keys
    unexpected_non_judge = [k for k in unexpected if not k.startswith("causal_judge.")]
    if unexpected_non_judge:
        logger.warning(f"Unexpected non-judge keys: {unexpected_non_judge}")
    
    logger.info(f"Generator loaded (skipped {len(unexpected)} causal_judge keys)")
    
    lit_module = lit_module.to(device)
    lit_module.eval()
    lit_module.freeze()

    logger.info(f"Generator: {lit_module.num_node_classes} node classes, "
                f"{lit_module.num_edge_classes} edge classes, "
                f"{lit_module.num_timesteps} timesteps")
    return lit_module



def sample_until_target(
    lit_module: DiGressModule,
    target_valid: int,
    batch_size: int = 50,
    guidance_scale: float = 0.0,
    node_count_range: tuple[int, int] = (10, 35),
    device: torch.device = torch.device("cuda"),
    output_path: str | Path | None = None,
    max_attempts: int = 200000,
) -> list[dict]:
    """Sample molecules in batches until we reach the target number of valid SMILES.

    Returns list of dicts with {smiles, SAS, qed, logP}.
    """
    valid_molecules = []
    seen_smiles = set()
    total_attempted = 0
    round_num = 0

    pbar = tqdm(total=target_valid, desc="Valid molecules", unit="mol")

    try:
        while len(valid_molecules) < target_valid and total_attempted < max_attempts:
            round_num += 1

            # Sample random node counts from a realistic distribution
            num_nodes_per_graph = np.random.randint(
                node_count_range[0], node_count_range[1] + 1, size=batch_size
            ).tolist()

            # Get a conditioning embedding
            c = lit_module.model.null_embedding.expand(batch_size, -1).to(device)

            try:
                X, E, edge_index, batch_idx = sample_digress_graphs(
                    model=lit_module.model,
                    c=c,
                    num_nodes_per_graph=num_nodes_per_graph,
                    node_marginals=lit_module.node_marginals,
                    edge_marginals=lit_module.edge_marginals,
                    num_node_classes=lit_module.num_node_classes,
                    num_edge_classes=lit_module.num_edge_classes,
                    num_timesteps=lit_module.num_timesteps,
                    guidance_scale=guidance_scale,
                )
            except Exception as e:
                logger.warning(f"Sampling batch failed: {e}")
                continue

            total_attempted += batch_size

            # Decode each graph to SMILES
            for i in range(batch_size):
                mask = (batch_idx == i)
                n_idx = torch.nonzero(mask).squeeze(-1)
                if len(n_idx) == 0:
                    continue

                node_types = X[mask]
                edge_mask = (batch_idx[edge_index[0]] == i)
                sub_edge_index = edge_index[:, edge_mask]
                offset = n_idx.min().item()
                sub_edge_index = sub_edge_index - offset
                edge_types = E[edge_mask]

                smiles = graph_to_smiles(node_types, sub_edge_index, edge_types)
                if smiles is None:
                    continue

                # Validate via RDKit roundtrip
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Canonicalize
                canon = Chem.MolToSmiles(mol)
                if canon in seen_smiles:
                    continue

                # Compute properties
                props = compute_properties(canon)
                if props is None:
                    continue

                seen_smiles.add(canon)
                props["smiles"] = canon  # Use canonical SMILES
                valid_molecules.append(props)
                pbar.update(1)

                if len(valid_molecules) >= target_valid:
                    break

            # Periodic saving and logging
            if round_num % 10 == 0:
                validity = len(valid_molecules) / max(total_attempted, 1) * 100
                logger.info(
                    f"Round {round_num}: {len(valid_molecules)}/{target_valid} valid "
                    f"({total_attempted} attempted, {validity:.1f}% validity)"
                )
                if output_path and len(valid_molecules) > 0:
                    pd.DataFrame(valid_molecules).to_csv(output_path, index=False)
                    logger.info(f"  (Periodic save to {output_path})")

            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving progress...")
    
    if total_attempted >= max_attempts:
        logger.warning(f"\nReached max attempts ({max_attempts}). Stopping.")

    pbar.close()

    validity = len(valid_molecules) / max(total_attempted, 1) * 100
    logger.info(f"\nFinal: {len(valid_molecules)} valid molecules from {total_attempted} attempts ({validity:.1f}% validity)")

    return valid_molecules



def main():
    parser = argparse.ArgumentParser(description="Sample molecules from DiGress generator")
    parser.add_argument("--clearml_id", type=str, default="152d4e4b1f4a42608f91d98d4fdf1b6f",
                        help="ClearML Model ID for the DiGress checkpoint")
    parser.add_argument("--target", type=int, default=20000,
                        help="Number of valid unique molecules to generate")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Molecules per sampling batch")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                        help="CFG guidance scale (0.0 = unconditional)")
    parser.add_argument("--min_nodes", type=int, default=10,
                        help="Minimum atoms per molecule")
    parser.add_argument("--max_nodes", type=int, default=35,
                        help="Maximum atoms per molecule")
    parser.add_argument("--output", type=str, default="data/raw/digress_generated.csv",
                        help="Output CSV path")
    parser.add_argument("--max_attempts", type=int, default=500000,
                        help="Maximum number of attempted molecules")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load generator
    lit_module = load_generator(args.clearml_id, device)

    # Sample
    t0 = time.time()
    molecules = sample_until_target(
        lit_module=lit_module,
        target_valid=args.target,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        node_count_range=(args.min_nodes, args.max_nodes),
        device=device,
        output_path=args.output,
        max_attempts=args.max_attempts,
    )
    elapsed = time.time() - t0
    if len(molecules) > 0:
        logger.info(f"Sampling took {elapsed/60:.1f} minutes ({elapsed/len(molecules):.2f}s per valid molecule)")
    else:
        logger.warning("No valid molecules sampled.")


    # Save
    df = pd.DataFrame(molecules)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\nSaved {len(df)} molecules to {output_path}")
    logger.info(f"Property statistics:")
    for col in ["SAS", "qed", "logP"]:
        logger.info(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, "
                     f"min={df[col].min():.3f}, max={df[col].max():.3f}")

    # Preview
    logger.info(f"\nSample molecules:")
    for _, row in df.head(5).iterrows():
        logger.info(f"  {row['smiles'][:40]}  SA={row['SAS']:.2f}  QED={row['qed']:.3f}  LogP={row['logP']:.2f}")


if __name__ == "__main__":
    main()
