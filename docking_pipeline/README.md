# Automated High-Throughput Docking Pipeline

This directory contains an end-to-end automated molecular docking pipeline. It handles 2D SMILES to 3D conversions, receptor preparation, AutoDock Vina execution, and PyMOL automated rendering.

## Setup

Use the `setup_docking.sh` script to install the required Python dependencies into your active `uv` environment.

```bash
chmod +x setup_docking.sh
./setup_docking.sh
```

*(Note: PyMOL is notoriously difficult to install cleanly via pure pip on some systems. If the `uv add pymol` fails or PyMOL complains about missing shared libraries during rendering, install it via `conda install -c conda-forge pymol-open-source` or your OS package manager.)*

## Step 1: Docking Execution

Run the orchestrator script. It requires an input CSV containing a `smiles` column and an `id` column.

```bash
python docking_pipeline.py --input example.csv --target 2VBC --out_dir docking_results
```

**What this does:**
1. Downloads `2VBC` from the RCSB PDB.
2. Cleans the protein (removes water) and extracts the native ligand's coordinate center.
3. Converts the protein to `.pdbqt` using `meeko`.
4. Converts your 2D SMILES into 3D, optimizes geometry (MMFF94), and assigns Gasteiger charges.
5. Runs AutoDock Vina for each ligand.
6. Saves the best pose and outputs affinities to `docking_results/docking_scores.csv`.

## Step 2: Visualization

Once docking is complete, you can generate high-resolution PyMOL renders for your hits.

```bash
python render_hits.py \
    --receptor docking_results/2VBC_clean.pdb \
    --ligand docking_results/ligand_1_docked.pdbqt \
    --out hit_render.png
```

This will automatically load the receptor and ligand, draw hydrogen bonds (yellow dashes), hide unnecessary clutter, and render a 1080p PNG.
