#!/usr/bin/env python3
"""Script to extract all SMILES from downloaded datasets and compute SAscores."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os

from rdkit import Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def main():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist. Please download some datasets first.")
        return

    # Gather all unique SMILES from all CSVs currently in data/raw
    all_smiles = set()
    
    # Try to find the SMILES column in any CSV
    for csv_file in raw_dir.glob("*.csv*"):
        if "sascore" in csv_file.name.lower():
            continue
            
        try:
            df = pd.read_csv(csv_file)
            # Try common SMILES column names
            smiles_cols = [c for c in df.columns if c.lower() in ["smiles", "mol", "drug"]]
            if smiles_cols:
                col = smiles_cols[0]
                smiles_list = df[col].dropna().astype(str).tolist()
                all_smiles.update(smiles_list)
                print(f"Extracted {len(smiles_list)} SMILES from {csv_file.name}")
        except Exception as e:
            print(f"Could not read {csv_file.name}: {e}")

    if not all_smiles:
        print("No SMILES found in any dataset. Please run the fine-tuning script or download_herg.py first.")
        return

    all_smiles = list(all_smiles)
    print(f"\nFound {len(all_smiles)} unique SMILES across all datasets.")
    print("Computing SAscore...")

    results = []
    for smi in tqdm(all_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                score = sascorer.calculateScore(mol)
                results.append({"smiles": smi, "sascore": score})
        except Exception:
            pass

    out_df = pd.DataFrame(results)
    out_path = raw_dir / "sascore.csv"
    out_df.to_csv(out_path, index=False)
    
    print(f"Successfully computed {len(out_df)} SAscores and saved to {out_path}")

if __name__ == "__main__":
    main()
