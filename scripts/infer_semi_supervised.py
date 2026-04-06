"""Inference script for the Semi-Supervised Molecular Joint Embedding model.

This script loads a pretrained multi-task model and predicts various biological 
and chemical properties (BBBP, ESOL, Tox21, etc.) for any given SMILES string.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch_geometric.data import Batch
from clearml import Model

# Local imports
from mol_prop_gnn.data.preprocessing import smiles_to_graph
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule
from mol_prop_gnn.models.factory import build_joint_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WELL_KNOWN_MOLECULES = {
    # --- WELL-KNOWN MOLECULES (Memorization Check) ---
    "Aspirin (Painkiller)": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (Stimulant)": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "Ibuprofen (Advil)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Dopamine (Neurotransmitter)": "c1cc(c(cc1CCN)O)O",
    "Glucose (Sugar)": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Nicotine (Stimulant)": "CN1CCCC1c2cccnc2",
    "Penicillin G (Antibiotic)": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",

    # --- MODERN DRUGS (Zero-Shot Generalization Check, Post-2019) ---
    "Nirmatrelvir (2021, Paxlovid)": "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(C#N)CC3CCNC3=O)C",
    "Lemborexant (2019, Insomnia)": "CC1=C(C=CC(=C1)C)C2=CC=CC(=C2)C(=O)NC3=NC=C(C=C3)C4=CCN(CC4)C(=O)C",
    "Lasmiditan (2019, Migraine)": "CN1CCC(CC1)C(=O)Nc2cccc(c2)C(=O)c3ccccc3F",
    "Cenobamate (2019, Epilepsy)": "NC(=O)OC(c1ccccc1Cl)c2nnnn2C",
    "Molnupiravir (2021, Antiviral)": "CC(C)C(=O)OCC1OC(n2ccc(=NO)nc2=O)C(O)C1O",
    "Bempedoic Acid (2020, Cholesterol)": "OC(=O)C(C)(C)CCCCCCCCC(O)C(C)(C)C(=O)O",
    
    # --- STRUCTURAL EXTREMES & TRAPS (Physics/Biology Stress-Test) ---
    "Super-Lipophilic Aspirin": "CCCCCCCC(=O)Oc1ccccc1C(=O)O",
    "Hexadecane (Pure Grease)": "CCCCCCCCCCCCCCCC",
    "Octane-octaol (Pure Sugar-Mimic)": "OCC(O)C(O)C(O)C(O)C(O)C(O)CO",
    "Perfluorohexane (Teflon-Mimic)": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "Permanent Charge Trap": "C[N+](C)(C)CCCCCCCCc1ccccc1",
    "Nitrogen Mustard Trap": "O=C(c1ccccc1)N(CCCl)CCCl",
    "Iron-Binding Trap": "O=c1c(O)c(-c2ccccc2)cn1C",

    "Tapinarof (2022, Psoriasis)": "CC(C)(C)c1cc(O)cc(C=Cc2ccccc2)c1",
    "Vonoprazan (2023, Acid Reflux)": "CNCC1=CC=CC=C1C2=CC=CN2S(=O)(=O)C3=CC=CC=C3F",
    "Gepirone (2023, Depression)": "CC1(CCC(=O)N(C1=O)CCCCN2CCN(CC2)c3ncccn3)C",
    "Mavacamten (2022, Heart Failure)": "CC(C)NC1=NC(=O)N(C(=O)C1)c2ccccc2",
    "Nirogacestat (2023, Tumor)": "CC(C)(C)NC(=O)C(Cc1ccccc1)NC(=O)C2(CCCC2)c3ccc(F)cc3F",
    "Viloxazine (2021, ADHD)": "CCOC1=CC=CC=C1OCC2CNCCO2",

    # --- 2. ENVIRONMENTAL TOXINS & TRAPS (Tox21 Master-Class) ---
    "TCDD (Dioxin - Ultimate AhR Trap)": "Clc1cc2Oc3cc(Cl)c(Cl)cc3Oc2cc1Cl",
    "Paraquat (Herbicide - MMP Trap)": "C[n+]1ccc(cc1)-c2cc[n+](C)cc2",
    "Bisphenol A (BPA - Estrogen Trap)": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
    "Diethylstilbestrol (DES - Estrogen Trap)": "CCC(=C(CC)c1ccc(O)cc1)c2ccc(O)cc2",
    "Aflatoxin B1 (Mold Toxin - p53 Trap)": "COC1=C2C3=C(C(=O)OCC3)C(=O)OC2=C4C5C=COC5OC4=C1",
    "MPTP (Synthetic Neurotoxin)": "CN1CCC(=CC1)c2ccccc2",
    "Thalidomide (Historic Teratogen)": "O=C1NC(=O)CCC1N2C(=O)c3ccccc3C2=O",

    # --- 3. PHYSICOCHEMICAL ODDITIES (ESOL / BBBP Stress Tests) ---
    "Cubane (Pure Geometric Stress)": "C12C3C4C1C5C4C3C25",
    "Squalene (Shark Oil - Extreme Lipophilicity)": "CC(=CCCC(=CCCC(=CCCC(=CCCC(=CCCC=C(C)C)C)C)C)C)C",
    "Mellitic Acid (Extreme Water Solubility)": "O=C(O)c1c(C(=O)O)c(C(=O)O)c(C(=O)O)c(C(=O)O)c1C(=O)O",
    "18-Crown-6 (Ionophore / Solvation Oddity)": "C1COCCOCCOCCOCCOCCO1",
    "Adamantane (3D Bulky Diamondoid)": "C1C2CC3CC1CC(C2)C3",
    "TNT (Explosive - Nitro Group Overload)": "Cc1c(N(=O)=O)cc(N(=O)=O)cc1N(=O)=O",

    # --- 4. THE HIV DATASET TARGET ---
    "Zidovudine (AZT - Classic HIV Inhibitor)": "CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)N=[N+]=[N-]",

    "Sucralose (Artificial Sweetener)": "OCC1OC(OC2(CCl)OC(CO)C(O)C2Cl)C(Cl)C(O)C1O",
    "Fexofenadine (Allegra - Safe)": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "Riboflavin (Vitamin B2)": "CC1=C(C)C=C2N(CC(O)C(O)C(O)CO)C3=NC(=O)NC(=O)C3=NC2=C1",
    "Vanillin (Vanilla Extract)": "COc1cc(C=O)ccc1O",

    # --- 2. THE "SILENT KILLERS" (Tiny, simple, but lethally toxic) ---
    # GNNs often ignore small molecules. Will it catch these massive Tox21 hazards?
    "Phosgene (WW1 Nerve Gas)": "O=C(Cl)Cl",
    "Acrolein (Toxic/Tear Gas)": "C=CC=O",
    "Fluoroacetate (Lethal Poison 1080)": "FCC(=O)O",
    "Formaldehyde (Carcinogen/Crosslinker)": "C=O",

    # --- 3. THE ENDOCRINE SYSTEM (Tox21 Nuclear Receptor Master-Class) ---
    # These should light up the NR-ER (Estrogen) and NR-AR (Androgen) targets perfectly.
    "Estradiol (Natural Estrogen)": "CC12CCC3C(CCC4=CC(O)=CC=C34)C1CCC2O",
    "Testosterone (Natural Androgen)": "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
    "Trenbolone (Synthetic Steroid)": "CC12CCC3C(=CCC4=CC(=O)CCC34)C1CCC2O",
    "Dexamethasone (Corticosteroid)": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",

    # --- 4. THE MODERN HIV/ANTIVIRAL BENCHMARK (2018-2020) ---
    # True zero-shot tests for your HIV dataset representation.
    "Cabotegravir (2020, HIV Integrase)": "CC1C2c3c(F)cc(F)cc3CN2C(=O)C4=C(O)C(=O)N(CC5=CC=C(F)C=C5)C=C4C1=O",
    "Doravirine (2018, HIV NNRTI)": "Cn1c(C(F)(F)F)cc(Cl)c2c1c(C#N)c(Oc3cc(C)nc(=O)n3C)cc2",
    "Bictegravir (2018, HIV Integrase)": "CC1CC2C(C1)N3C(=O)C4=C(O)C(=O)N(CC5=C(F)C=CC(=C5)F)C=C4C3=C2O",
    "GS-441524 (Remdesivir Metabolite)": "N#CC1(C2=CC=NN2C3=NC=NN3C1)C(O)C(O)CO",

    # --- 5. SEVERE AGROCHEMICALS (Tox21 Environmental Hazards) ---
    # These should heavily trigger SR-MMP (Mitochondrial) and SR-ARE (Oxidative Stress).
    "Chlorpyrifos (Organophosphate)": "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",
    "Glyphosate (Roundup Herbicide)": "O=C(O)CNCP(=O)(O)O",
    "Atrazine (Herbicide/Endocrine)": "CCNC1=NC(=NC(=N1)Cl)NC(C)C",
    "Permethrin (Insecticide)": "CC1(C)C(C1C=C(Cl)Cl)C(=O)OCC2=CC=CC(=C2)OC3=CC=CC=C3"
}

def load_model(checkpoint_path: str):
    """Reconstructs the model and loads weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    
    # 1. Get model config from hparams (Stage 2 checkpoints will have this)
    model_config = hparams.get("model_config")
    
    # 2. Fallback for Stage 1 checkpoints (like version_5) that don't have model_config yet
    if not model_config:
        from mol_prop_gnn.data.preprocessing import get_node_feature_dim, get_edge_feature_dim
        logger.warning("No 'model_config' found in checkpoint. Falling back to GCN defaults.")
        # We know version_5 was: GCN, 13 tasks, 256 bottleneck, 0.3 dropout
        model_config = {
            "backbone_name": "gcn",
            "node_dim": get_node_feature_dim(), # Should be 38
            "edge_dim": get_edge_feature_dim(), # Should be 13
            "num_tasks": len(hparams.get("dataset_names", [])),
            "bottleneck_dim": 256,
            "dropout": 0.3,
            "deg": None
        }
    
    # 3. Reconstruct base model
    model = build_joint_model(**model_config)
    
    # 4. Load weights into Lightning Module
    lit_model = JointSemiSupModule.load_from_checkpoint(
        checkpoint_path, 
        model=model, 
        map_location="cpu"
    )
    lit_model.eval()
    return lit_model

def run_inference(checkpoint_path: str, smiles_list: list[str], molecule_names: list[str] | None = None):
    """Loads a model and runs inference on a list of SMILES."""
    
    # 1. Load Model
    logger.info("Loading model from: %s", checkpoint_path)
    model = load_model(checkpoint_path)
    
    # 2. Extract Metadata from Model Hyperparameters
    task_names = model.hparams.dataset_names
    task_types = model.hparams.task_types
    
    # 3. Process Molecules
    graph_list = []
    valid_names = []
    
    for i, smiles in enumerate(smiles_list):
        name = molecule_names[i] if molecule_names else f"Molecule {i+1}"
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph_list.append(graph)
            valid_names.append(name)
        else:
            logger.warning("Failed to convert SMILES: %s", smiles)

    if not graph_list:
        logger.error("No valid molecules to process.")
        return

    # 4. Run Inference
    batch = Batch.from_data_list(graph_list)
    
    print("\n" + "="*60)
    print(" 🧪 MOLECULAR PROPERTY PREDICTIONS")
    print("="*60)

    with torch.no_grad():
        # Predictions shape: (num_molecules, num_tasks)
        logits = model(batch)
        
        for mol_idx, mol_name in enumerate(valid_names):
            print(f"\n🔹 {mol_name}")
            print("-" * 40)
            
            for task_idx, (task_name, task_type) in enumerate(zip(task_names, task_types)):
                val = logits[mol_idx, task_idx].item()
                
                if task_type == "classification":
                    prob = torch.sigmoid(torch.tensor(val)).item()
                    # Color coding probability (rough console approximation)
                    indicator = "✅" if prob > 0.5 else "❌"
                    print(f"  {indicator} {task_name:<20}: {prob:>6.1%} confidence")
                else:
                    # Regression
                    print(f"  📊 {task_name:<20}: {val:>8.3f} (scaled)")
                    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Molecular Multi-Task Inference")
    parser.add_argument("--checkpoint", type=str, help="Path to local .ckpt file")
    parser.add_argument("--model_id", type=str, help="ClearML Model ID to download weights from")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to predict")
    args = parser.parse_args()

    # Determine checkpoint source
    final_path = args.checkpoint
    if args.model_id:
        logger.info("Fetching model from ClearML: %s", args.model_id)
        final_path = Model(model_id=args.model_id).get_local_copy()

    if not final_path or not Path(final_path).exists():
        logger.error("No valid checkpoint provided. Use --checkpoint or --model_id")
        return

    # Determine input molecules
    if args.smiles:
        smiles_list = [args.smiles]
        molecule_names = ["Custom Query"]
    else:
        logger.info("No SMILES provided. Running sanity check on well-known drugs...")
        smiles_list = list(WELL_KNOWN_MOLECULES.values())
        molecule_names = list(WELL_KNOWN_MOLECULES.keys())

    run_inference(final_path, smiles_list, molecule_names)

if __name__ == "__main__":
    main()
