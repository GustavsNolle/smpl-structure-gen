"""Download the CEPII Gravity Database and BACI.

The CEPII Gravity dataset provides bilateral trade flows at the
country-pair-year level together with standard gravity covariates.

Reference: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8
Reference for BACI: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Default URL for the CEPII Gravity CSV (V202211 release)
DEFAULT_URL = (
    "http://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"
)

# Default URL for BACI HS92 (V202401b release)
DEFAULT_BACI_URL = (
    "http://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS92_V202401b.zip"
)

def download_gravity_data(
    url: str = DEFAULT_URL,
    raw_dir: str | Path = "data/raw",
    *,
    force: bool = False,
) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = list(raw_dir.glob("Gravity_V*.csv"))
    if csv_candidates and not force:
        logger.info("CEPII Gravity data already exists: %s", csv_candidates[0])
        return csv_candidates[0]

    logger.info("Downloading CEPII Gravity data from %s ...", url)
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    total_bytes = 0
    content = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        content.write(chunk)
        total_bytes += len(chunk)
        logger.info("  Downloaded %.1f MB ...", total_bytes / 1e6)

    content.seek(0)
    with zipfile.ZipFile(content) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV files found in the downloaded archive.")
        for csv_name in csv_names:
            logger.info("Extracting %s ...", csv_name)
            zf.extract(csv_name, raw_dir)

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {raw_dir}")
        
    extracted = max(csv_files, key=lambda p: p.stat().st_size)
    logger.info("Main CEPII Gravity data identified by size (%.1f MB) at %s", 
                extracted.stat().st_size / 1e6, extracted)
    return extracted

def download_baci_data(
    url: str = DEFAULT_BACI_URL,
    raw_dir: str | Path = "data/raw",
    *,
    force: bool = False,
) -> Path:
    raw_dir = Path(raw_dir)
    baci_dir = raw_dir / "baci"
    baci_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "BACI_temp_download.zip"

    # Check if already extracted
    csv_candidates = list(baci_dir.glob("country_codes_V*.csv"))
    if csv_candidates and not force:
        logger.info("CEPII BACI data already extracted in %s", baci_dir)
        return baci_dir

    # Resume logic
    headers = {}
    mode = 'ab'
    initial_size = 0
    if zip_path.exists() and not force:
        initial_size = zip_path.stat().st_size
        headers['Range'] = f'bytes={initial_size}-'
        logger.info("Found partial download of %.1f MB. Resuming...", initial_size / 1e6)
    elif force and zip_path.exists():
        zip_path.unlink()
        mode = 'wb'
    else:
        mode = 'wb'

    logger.info("Downloading CEPII BACI data from %s ...", url)
    try:
        response = requests.get(url, headers=headers, timeout=300, stream=True)
        
        if response.status_code == 416:
            logger.info("File already fully downloaded.")
        else:
            response.raise_for_status()
            
            # If server ignores range header and returns 200, we must restart
            if response.status_code == 200 and initial_size > 0:
                logger.warning("Server does not support resume. Restarting download.")
                mode = 'wb'
                initial_size = 0
                
            with open(zip_path, mode) as f:
                total_bytes = initial_size
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
                        # Log roughly every 50MB
                        if total_bytes % (50 * 1024 * 1024) < 1024 * 1024:
                            logger.info("  Downloaded %.1f MB ...", total_bytes / 1e6)
                            
        logger.info("Download completed successfully! Extracting BACI archive to %s ...", baci_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(baci_dir)

        logger.info("BACI data extracted successfully to %s", baci_dir)
        
        # Clean up the massive zip file to save disk space
        zip_path.unlink(missing_ok=True)
        return baci_dir
        
    except requests.exceptions.ChunkedEncodingError as e:
        logger.error("Connection interrupted! Data is safely saved to disk. Just rerun the script to resume: %s", e)
        # We don't delete the zip_path so it can be resumed
        raise
