# MolPropGNN

**Small Molecule Property Prediction with Graph Neural Networks**

## Research Question

> Can graph neural networks that operate directly on molecular graph structure predict molecular properties better than classical fingerprint-based ML baselines?
>
> *Supporting question:* Does explicit message passing over atomic connectivity improve prediction beyond global molecular descriptors alone?

## Overview

This project represents small molecules as **molecular graphs** where:
- **Nodes** = atoms (with features: atomic number, degree, hybridization, aromaticity, etc.)
- **Edges** = chemical bonds (with features: bond type, stereo, conjugation, ring membership)
- **Graph-level labels** = molecular properties (solubility, permeability, toxicity, etc.)

The model predicts **molecular properties** from the MoleculeNet benchmark using:
- Atom-level features derived from RDKit
- Bond-level features
- Graph neural network architectures (GCN, GAT, EGNN, GINE, RGCN)

## Project Structure

```
mol_prop_gnn/
├── configs/              # YAML configuration files
├── data/                 # Raw and processed data (gitignored)
├── scripts/              # Entry-point scripts
│   ├── train.py          # Main training script
│   ├── evaluate_models.py # Model comparison
│   └── run_experiments.py # Multi-dataset benchmark
├── src/smpl-structure-gen/  # Main source package
│   ├── data/             # Data loading, SMILES→graph conversion
│   ├── models/           # GCN, GAT, EGNN, GINE, RGCN, baselines
│   ├── training/         # PyTorch Lightning module
│   ├── evaluation/       # Classification & regression metrics
│   └── utils/            # Configuration utilities
└── tests/                # Unit & smoke tests
```

## Supported Datasets (MoleculeNet)

| Dataset | Task | # Molecules | Metric |
|---------|------|-------------|--------|
| **BBBP** | Classification | 2,039 | AUROC |
| **ESOL** | Regression | 1,128 | RMSE |
| BACE | Classification | 1,513 | AUROC |
| FreeSolv | Regression | 642 | RMSE |
| Lipophilicity | Regression | 4,200 | RMSE |
| HIV | Classification | 41,127 | AUROC |
| Tox21 | Multi-task Classification | 7,831 | AUROC |

## Model Benchmark

| Model | Type | Graph Structure? |
|-------|------|-----------------|
| RDKit (RF) | Random Forest on descriptors | ✗ |
| XGBoost (FP) | XGBoost on Morgan fingerprints | ✗ |
| LightGBM (FP) | LightGBM on Morgan fingerprints | ✗ |
| MLP Baseline | MLP on pooled atom features | ✗ |
| **MolGCN** | GINE-based GNN | ✓ |
| **MolGAT** | GATv2 attention GNN | ✓ |
| **MolEGNN** | Edge-conditioned GNN | ✓ |
| **MolGINE** | GINE + ReZero hybrid | ✓ |
| **MolRGCN** | Relational GCN (bond-typed) | ✓ |

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package and environment management.

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

## Quick Start

```bash
# 1. Train the GCN model on BBBP (auto-downloads dataset)
uv run python scripts/train.py --config configs/default.yaml

# 2. Train a specific model
uv run python scripts/train.py --config configs/default.yaml --model gat

# 3. Run full multi-model benchmark
uv run python scripts/run_experiments.py

# 4. Evaluate all trained models
uv run python scripts/evaluate_models.py

# 5. Run tests
uv run pytest tests/ -v
```

## Data Source

**MoleculeNet** — A benchmark for molecular machine learning providing standardized datasets with scaffold-based splitting for realistic evaluation.

Reference: Wu et al., "MoleculeNet: A Benchmark for Molecular Machine Learning", *Chemical Science*, 2018.

## Tech Stack

- **PyTorch** + **PyTorch Geometric** — GNN implementation
- **PyTorch Lightning** — training loop, logging, checkpointing
- **RDKit** — molecular graph construction & featurization
- **scikit-learn** — baseline models & metrics
- **XGBoost** / **LightGBM** — gradient boosted tree baselines

## License

MIT
