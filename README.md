# SMPL Structure Gen

**Anchor-Guided 3D Molecule Generation via Causal DiGress & Automated Docking**

## Overview

This repository provides an end-to-end pipeline for generating novel, drug-like 3D molecular structures conditionally guided by anchor molecules. It transitions from traditional property prediction (MolPropGNN) to a full generative pipeline utilizing Graph Neural Networks, Causal Learning, and Discrete Denoising Diffusion (DiGress). 

The generated molecules are evaluated using lightweight "Mini Judges" (for SA Score, QED, LogP) and validated through an automated AutoDock Vina pipeline.

## Project Structure

```text
smpl-structure-gen/
├── configs/              # YAML configuration files for training
├── data/                 # Raw and processed data (ZINC250K, MoleculeNet)
├── docking_pipeline/     # Automated Vina docking and PyMOL rendering scripts
├── scripts/              # Training and inference entry-points
│   ├── train_pretrain_masked.py  # Phase 1: Self-supervised pre-training
│   ├── train_causal.py           # Phase 1: Causal Semi-supervised mapping
│   ├── train_mini_judge.py       # Phase 2: Property Mini Judges
│   ├── train_digress.py          # Phase 3: Causal DiGress generation model
│   └── inference_guided.py       # Anchor-guided generation inference
├── src/mol_prop_gnn/     # Main source package containing models, data, and utils
└── tests/                # Unit tests
```

## Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Install uv

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Project Setup

```bash
# Clone the repository
git clone <repo-url>
cd smpl-structure-gen

# Install dependencies and create environment
uv sync --all-extras
```

### 3. Docking Dependencies

To use the automated docking and rendering pipeline, install additional PyMOL dependencies. (You may also need to install PyMOL via conda or OS package manager if pip installation fails).

```bash
cd docking_pipeline
chmod +x setup_docking.sh
./setup_docking.sh
cd ..
```

## Pipeline & Replication Instructions

The workflow is divided into three training phases, followed by generation and docking validation.

### Phase 1: Pre-training & Causal Mapping
Train the base Graph Neural Network representations using masked node prediction and causal learning on molecular datasets.

```bash
# Self-supervised continuous masked node prediction on ZINC
uv run python scripts/train_pretrain_masked.py

# Map Semi-Supervised Training via Graph Causal Learning
uv run python scripts/train_causal.py --config configs/causal.yaml
```

### Phase 2: Mini Judges
Train lightweight GIN models to act as property evaluators (SA Score, QED, LogP) during the diffusion process.

```bash
# Train all property judges (sascore, qed, logp)
uv run python scripts/train_mini_judge.py --property all
```

### Phase 3: Causal DiGress (Diffusion)
Train the Discrete Denoising Diffusion model, guided by the frozen Causal Judge from Phase 1.

```bash
uv run python scripts/train_digress.py --config configs/digress.yaml
```

### Inference: Anchor-Guided Generation
Generate new molecules conditionally guided by specific anchor molecules (e.g., FDA-approved drugs or clinical failures). Generates molecules and creates image grids.

```bash
uv run python scripts/inference_guided.py --num_samples 500 --batch_size 50
```
*Outputs will be saved to the `outputs/` directory, including CSVs of valid/filtered molecules and grid images.*

### Docking Validation
Dock the generated molecules against a target receptor (e.g., `2VBC`) using AutoDock Vina.

```bash
# Run the docking orchestrator
uv run python docking_pipeline/docking_pipeline.py --input outputs/filtered_molecules.csv --target 2VBC --out_dir docking_results

# Render the best hits using PyMOL
uv run python docking_pipeline/render_hits.py --receptor docking_results/2VBC_clean.pdb --ligand docking_results/ligand_1_docked.pdbqt --out hit_render.png
```

## Tech Stack

- **PyTorch & PyTorch Geometric** — GNN backbone and Diffusion implementation
- **PyTorch Lightning & ClearML** — Training loop, distributed training, and tracking
- **RDKit** — Molecular graph construction, SA Score, QED, and 2D→3D conversion
- **Meeko & AutoDock Vina** — Ligand preparation and automated molecular docking
- **PyMOL** — Automated high-resolution pose rendering

## License

MIT
