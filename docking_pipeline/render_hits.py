#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# PyMOL headless init
import pymol
pymol.pymol_argv = ['pymol', '-qc'] # Quiet and headless mode
pymol.finish_launching()
from pymol import cmd

def render_pose(receptor_pdb: str, ligand_pdbqt: str, out_png: str):
    """Renders the receptor and ligand, highlights polar contacts, and saves PNG."""
    
    # 1. Initialization
    cmd.reinitialize()
    cmd.bg_color("white")
    
    # 2. Load objects
    cmd.load(receptor_pdb, "receptor")
    cmd.load(ligand_pdbqt, "ligand")
    
    # 3. Receptor styling (Surface, transparent gray)
    cmd.hide("everything", "receptor")
    cmd.show("surface", "receptor")
    cmd.color("gray70", "receptor")
    cmd.set("transparency", 0.4, "receptor")
    
    # 4. Ligand styling (Sticks, brightly colored)
    cmd.hide("everything", "ligand")
    cmd.show("sticks", "ligand")
    cmd.color("cyan", "ligand and elem C")
    
    # Optional: If you want magenta instead of cyan, uncomment below
    # cmd.color("magenta", "ligand and elem C")
    
    # 5. Polar Contacts (Hydrogen Bonds)
    # distance <name>, <sel1>, <sel2>, [cutoff], [mode]
    # mode=2 means polar contacts only
    cmd.distance("hbonds", "receptor", "ligand", cutoff=3.2, mode=2)
    cmd.hide("labels", "hbonds") # Hide distance text labels to keep it clean
    cmd.set("dash_color", "yellow", "hbonds")
    cmd.set("dash_width", 3.0, "hbonds")
    
    # 6. Camera setup
    # Focus on ligand and zoom out slightly
    cmd.zoom("ligand", buffer=4.0)
    cmd.orient("ligand")
    
    # 7. Ray-tracing and Output
    print(f"Ray-tracing and saving {out_png}...")
    # dpi=300, 1920x1080 resolution
    cmd.png(out_png, width=1920, height=1080, dpi=300, ray=1)
    
def main():
    parser = argparse.ArgumentParser(description="Automated PyMOL rendering of docking hits.")
    parser.add_argument("--receptor", required=True, help="Path to cleaned receptor PDB (e.g. 2VBC_clean.pdb)")
    parser.add_argument("--ligand", required=True, help="Path to docked ligand PDBQT (e.g. ligand_1_docked.pdbqt)")
    parser.add_argument("--out", required=True, help="Output PNG path (e.g. hit_render.png)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.receptor):
        print(f"Error: Receptor file {args.receptor} not found.")
        sys.exit(1)
    if not os.path.exists(args.ligand):
        print(f"Error: Ligand file {args.ligand} not found.")
        sys.exit(1)
        
    render_pose(args.receptor, args.ligand, args.out)
    print("Render complete!")

if __name__ == "__main__":
    main()
