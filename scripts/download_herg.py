#!/usr/bin/env python3
"""Script to download the hERG dataset using Therapeutics Data Commons (PyTDC)."""

import pandas as pd
from pathlib import Path
import sys
import subprocess

def main():
    try:
        from tdc.single_pred import Tox
    except ImportError:
        print("PyTDC is not installed. Installing it now...")
        subprocess.check_call(["uv", "pip", "install", "PyTDC"])
        from tdc.single_pred import Tox

    print("Downloading hERG dataset via PyTDC...")
    # Fetch the hERG dataset (specifically hERG_Karim is standard, but 'hERG' defaults to a curated subset)
    # The default 'hERG' is broken upstream in PyTDC, so we use 'hERG_Karim' which works perfectly.
    data = Tox(name='hERG_Karim')
    df = data.get_data()
    
    # TDC usually returns 'Drug_ID', 'Drug' (SMILES), 'Y' (label)
    # Rename them so they map perfectly into our pipeline
    if 'Drug' in df.columns:
        df = df.rename(columns={'Drug': 'smiles'})
    if 'Y' in df.columns:
        df = df.rename(columns={'Y': 'hERG'})
        
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "herg.csv"
    
    df.to_csv(out_path, index=False)
    print(f"Successfully saved hERG dataset with {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()
