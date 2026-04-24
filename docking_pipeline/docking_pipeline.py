#!/usr/bin/env python3
import argparse
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# RDKit for Ligand 2D -> 3D
from rdkit import Chem
from rdkit.Chem import AllChem

# Meeko for PDBQT conversion
from meeko import MoleculePreparation

# AutoDock Vina Python bindings
from vina import Vina

# Biopython for PDB fetching and parsing
from Bio.PDB import PDBList, PDBParser, Select, PDBIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NonWaterSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() not in ["HOH", "WAT"]

def fetch_and_clean_receptor(pdb_id: str, out_dir: Path, native_resname: str = None):
    """Fetches PDB, removes water, extracts ligand center, and saves clean PDB."""
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=str(out_dir), file_format="pdb")
    
    # Biopython sometimes saves as .ent, let's rename it if needed
    if pdb_file.endswith(".ent"):
        new_path = out_dir / f"{pdb_id}.pdb"
        os.rename(pdb_file, new_path)
        pdb_file = str(new_path)
        
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    
    # Clean structure (remove waters)
    clean_pdb_path = out_dir / f"{pdb_id}_clean.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(clean_pdb_path), NonWaterSelect())
    
    # Find grid center
    center = None
    ligand_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # If native_resname is provided, use it. Otherwise, look for large HETATMs.
                resname = residue.get_resname().strip()
                if resname not in ["HOH", "WAT"] and residue.id[0] != " ":
                    # It's a HETATM
                    if native_resname and resname == native_resname:
                        for atom in residue:
                            ligand_coords.append(atom.get_coord())
                    elif not native_resname:
                        # Heuristic: collect coords of all HETATMs (excluding ions maybe, but let's keep it simple)
                        for atom in residue:
                            ligand_coords.append(atom.get_coord())

    if ligand_coords:
        center = np.mean(ligand_coords, axis=0).tolist()
        logger.info(f"Found ligand/HETATM cluster. Center of Mass: {center}")
    else:
        logger.warning(f"No native ligand found. Defaulting center to [0, 0, 0].")
        center = [0.0, 0.0, 0.0]
        
    return clean_pdb_path, center

def prep_receptor_pdbqt(clean_pdb_path: Path) -> Path:
    """Converts a clean PDB to PDBQT format manually.
    Vina's default scoring function ignores receptor partial charges, 
    so we append 0.000 and simple AutoDock atom types to satisfy the format.
    """
    receptor_pdbqt_path = clean_pdb_path.with_suffix(".pdbqt")
    
    with open(clean_pdb_path, 'r') as f_in, open(receptor_pdbqt_path, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = line[76:78].strip()
                if not element: 
                    element = line[12:16].strip()[0] # fallback to atom name
                
                # Basic AutoDock type mapping
                ad_type = element
                if element == "N": ad_type = "N"
                elif element == "O": ad_type = "OA"
                elif element == "S": ad_type = "S"
                elif element == "H": ad_type = "HD"
                elif element == "P": ad_type = "P"
                elif element == "C": ad_type = "C"
                
                # Pad to 66 chars
                # Cols 67-70: blank ("    ")
                # Cols 71-76: charge (" 0.000")
                # Col 77: blank (" ")
                # Cols 78-79: AD type
                padded_line = line[:66].ljust(66)
                new_line = padded_line + "    " + " 0.000" + " " + f"{ad_type:<2}\n"
                f_out.write(new_line)
            elif line.startswith("TER") or line.startswith("END"):
                f_out.write(line)
                
    logger.info(f"Receptor converted to PDBQT: {receptor_pdbqt_path}")
    return receptor_pdbqt_path

def prep_ligand_pdbqt(smiles: str, name: str, out_dir: Path) -> str:
    """Converts 2D SMILES -> 3D RDKit -> Meeko PDBQT string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {name}: {smiles}")
        
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    prep = MoleculePreparation()
    prep.prepare(mol)
    pdbqt_string = prep.write_pdbqt_string()
    
    # Save to file for debugging/records
    pdbqt_path = out_dir / f"{name}.pdbqt"
    with open(pdbqt_path, "w") as f:
        f.write(pdbqt_string)
        
    return pdbqt_string

def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt_str: str, center: list, out_path: str):
    """Executes AutoDock Vina."""
    v = Vina(sf_name='vina')
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt_str)
    
    # 20x20x20 Box
    v.compute_vina_maps(center=center, box_size=[20.0, 20.0, 20.0])
    
    # Dock
    v.dock(exhaustiveness=8, n_poses=1)
    
    # Get energy
    energies = v.energies(n_poses=1)
    best_affinity = energies[0][0] if energies is not None and len(energies) > 0 else 0.0
    
    # Save best pose
    v.write_poses(out_path, n_poses=1, overwrite=True)
    
    return best_affinity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV containing 'smiles' and 'id' columns.")
    parser.add_argument("--target", default="2VBC", help="PDB ID of the target receptor.")
    parser.add_argument("--native_ligand", default=None, help="Resname of native ligand for centering (e.g. NAG).")
    parser.add_argument("--out_dir", default="docking_results", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Prepare Receptor
    logger.info("=== STEP 1: RECEPTOR PREPARATION ===")
    clean_pdb, grid_center = fetch_and_clean_receptor(args.target, out_dir, args.native_ligand)
    receptor_pdbqt = prep_receptor_pdbqt(clean_pdb)

    # Load input SMILES
    df = pd.read_csv(args.input)
    if 'smiles' not in df.columns:
        # Try without headers if smiles isn't found
        df = pd.read_csv(args.input, header=None)
        df.rename(columns={0: 'smiles'}, inplace=True)
        
    if 'id' not in df.columns:
        df['id'] = [f"ligand_{i}" for i in range(len(df))]

    results = []

    # 2 & 3. Prepare Ligands & Dock
    logger.info("=== STEP 2 & 3: LIGAND PREP & VINA DOCKING ===")
    for idx, row in df.iterrows():
        smi = row['smiles']
        lid = row['id']
        logger.info(f"Processing {lid}...")
        
        try:
            lig_pdbqt_str = prep_ligand_pdbqt(smi, lid, out_dir)
            lig_in_path = out_dir / f"{lid}.pdbqt"
            pose_out = out_dir / f"{lid}_docked.pdbqt"
            
            affinity = run_vina_docking(str(receptor_pdbqt), lig_pdbqt_str, grid_center, str(pose_out))
            
            logger.info(f"  -> Best Affinity: {affinity:.2f} kcal/mol")
            
            if affinity <= -5.0:
                row_dict = row.to_dict()
                row_dict["affinity"] = affinity
                row_dict["pose_file"] = str(pose_out)
                results.append(row_dict)
            else:
                # Cleanup files if threshold isn't met
                if pose_out.exists(): pose_out.unlink()
                if lig_in_path.exists(): lig_in_path.unlink()
                
        except Exception as e:
            logger.error(f"Failed docking for {lid}: {e}")
            # Cleanup on failure
            lig_in_path = out_dir / f"{lid}.pdbqt"
            pose_out = out_dir / f"{lid}_docked.pdbqt"
            if pose_out.exists(): pose_out.unlink()
            if lig_in_path.exists(): lig_in_path.unlink()

    # Save results
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by="affinity", ascending=True) # Lowest is best
    res_csv = out_dir / "docking_scores.csv"
    res_df.to_csv(res_csv, index=False)
    logger.info(f"Done! Results saved to {res_csv}")

if __name__ == "__main__":
    main()
