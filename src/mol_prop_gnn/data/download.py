"""Download MoleculeNet benchmark datasets.

Supports downloading standard MoleculeNet datasets (BBBP, ESOL, BACE,
FreeSolv, Tox21, HIV, etc.) from public sources.

Uses PyTorch Geometric's built-in MoleculeNet loader when available,
with a fallback to direct CSV download from DeepChem.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Direct download URLs for common MoleculeNet datasets (DeepChem hosted)
MOLECULENET_URLS = {
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}

# Column mappings: (smiles_column, target_columns, task_type)
DATASET_META = {
    "bbbp": ("smiles", ["p_np"], "classification"),
    "esol": ("smiles", ["measured log solubility in mols per litre"], "regression"),
    "freesolv": ("smiles", ["expt"], "regression"),
    "lipophilicity": ("smiles", ["exp"], "regression"),
    "bace": ("mol", ["Class"], "classification"),
    "hiv": ("smiles", ["HIV_active"], "classification"),
    "tox21": ("smiles", [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ], "classification"),
}


def download_moleculenet(
    dataset_name: str,
    raw_dir: str | Path = "data/raw",
    *,
    force: bool = False,
) -> Path:
    """Download a MoleculeNet dataset CSV.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (bbbp, esol, freesolv, etc.).
    raw_dir : str or Path
        Directory to store the downloaded file.
    force : bool
        If True, re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    import requests

    dataset_name = dataset_name.lower()
    if dataset_name not in MOLECULENET_URLS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(MOLECULENET_URLS.keys())}"
        )

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = MOLECULENET_URLS[dataset_name]
    filename = url.split("/")[-1]
    output_path = raw_dir / filename

    if output_path.exists() and not force:
        logger.info("Dataset %s already exists at %s", dataset_name, output_path)
        return output_path

    logger.info("Downloading %s from %s ...", dataset_name, url)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    logger.info(
        "Downloaded %s (%.2f MB) to %s",
        dataset_name,
        total_bytes / 1e6,
        output_path,
    )
    return output_path


def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata for a MoleculeNet dataset.

    Returns
    -------
    dict with keys: smiles_col, target_cols, task_type
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_META:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    smiles_col, target_cols, task_type = DATASET_META[dataset_name]
    return {
        "smiles_col": smiles_col,
        "target_cols": target_cols,
        "task_type": task_type,
        "num_tasks": len(target_cols),
    }
