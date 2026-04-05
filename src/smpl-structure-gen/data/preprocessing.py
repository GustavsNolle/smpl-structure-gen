"""Preprocessing pipeline for the CEPII Gravity and BACI datasets.

Transforms raw CSVs into per-year graph-ready DataFrames with node features,
edge features (from Gravity), and log-trade targets (from BACI or Gravity).
If BACI is used, yields a multigraph where each row has a specific 'edge_type'.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Default column mappings ───────────────────────────────────────────────

_REQUIRED_COLS = [
    "year",
    "iso3_o",        
    "iso3_d",        
    "tradeflow_comtrade_o",  
    "gdp_o",
    "gdp_d",
    "gdpcap_o",
    "gdpcap_d",
    "pop_o",
    "pop_d",
    "distw_harmonic",  
    "contig",        
    "comlang_off",   
    "col_dep_ever",  
    "comrelig",      
]

_OPTIONAL_COLS = [
    "fta_wto",
]

def get_hs_section(hs6_code: int) -> int:
    """Map a 6-digit HS product code to one of 21 broad HS Sections."""
    # Ensure it's string and at least 6 characters by zero-padding, then take first 2
    hs2 = int(str(hs6_code).zfill(6)[:2])
    
    if 1 <= hs2 <= 5: return 0
    elif 6 <= hs2 <= 14: return 1
    elif 15 <= hs2 <= 15: return 2
    elif 16 <= hs2 <= 24: return 3
    elif 25 <= hs2 <= 27: return 4
    elif 28 <= hs2 <= 38: return 5
    elif 39 <= hs2 <= 40: return 6
    elif 41 <= hs2 <= 43: return 7
    elif 44 <= hs2 <= 46: return 8
    elif 47 <= hs2 <= 49: return 9
    elif 50 <= hs2 <= 63: return 10
    elif 64 <= hs2 <= 67: return 11
    elif 68 <= hs2 <= 70: return 12
    elif 71 <= hs2 <= 71: return 13
    elif 72 <= hs2 <= 83: return 14
    elif 84 <= hs2 <= 85: return 15
    elif 86 <= hs2 <= 89: return 16
    elif 90 <= hs2 <= 92: return 17
    elif 93 <= hs2 <= 93: return 18
    elif 94 <= hs2 <= 96: return 19
    elif 97 <= hs2 <= 99: return 20
    else: return 0  # Default or other


def get_config_hash(config: dict[str, Any]) -> str:
    relevant_keys = [
        "countries", "year_start", "year_end", 
        "edge_features", "node_features", "use_baci", "use_deltas", "num_lags"
    ]
    subset = {k: config.get(k) for k in relevant_keys if k in config}
    
    if "countries" in subset:
        subset["countries"] = sorted(subset["countries"])
    if "edge_features" in subset:
        subset["edge_features"] = sorted(subset["edge_features"])
    if "node_features" in subset:
        subset["node_features"] = sorted(subset["node_features"])
        
    config_str = json.dumps(subset, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def load_and_filter(
    csv_path: str | Path,
    countries: list[str],
    year_start: int = 2000,
    year_end: int = 2019,
) -> pd.DataFrame:
    logger.info("Loading raw Gravity data from %s ...", csv_path)

    cols_to_use = list(_REQUIRED_COLS)
    sample = pd.read_csv(csv_path, nrows=5)
    for col in _OPTIONAL_COLS:
        if col in sample.columns:
            cols_to_use.append(col)

    chunk_list = []
    chunksize = 100_000
    country_set = set(countries)
    
    logger.info("Reading CSV in chunks of %d rows...", chunksize)
    reader = pd.read_csv(csv_path, usecols=cols_to_use, chunksize=chunksize, low_memory=False)
    
    pbar = tqdm(desc="Preprocessing Gravity data", unit="chunk")
    
    for i, chunk in enumerate(reader):
        pbar.update(1)
        chunk = chunk[(chunk["year"] >= year_start) & (chunk["year"] <= year_end)]
        if chunk.empty:
            continue
            
        chunk = chunk[chunk["iso3_o"].isin(country_set) & chunk["iso3_d"].isin(country_set)]
        if chunk.empty:
            continue

        chunk = chunk[chunk["iso3_o"] != chunk["iso3_d"]]
        
        # Keep rows even if tradeflow_comtrade_o is missing IF we're using BACI later
        chunk = chunk.dropna(subset=["distw_harmonic"])
        
        if not chunk.empty:
            chunk_list.append(chunk)

    pbar.close()

    if not chunk_list:
        logger.error("No data found after filtering! Check your country list and year range.")
        return pd.DataFrame(columns=cols_to_use)

    df = pd.concat(chunk_list, ignore_index=True)
    logger.info("Filtered Gravity data shape: %s", df.shape)
    return df

def load_and_merge_baci(df_grav: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    baci_dir = raw_dir / "baci"
    
    if not baci_dir.exists():
        logger.warning(f"BACI directory {baci_dir} not found. Proceeding without BACI.")
        df_grav["edge_type"] = 0
        df_grav["log_trade"] = np.log1p(df_grav["tradeflow_comtrade_o"].fillna(0))
        return df_grav

    # Load country codes mapping
    country_files = list(baci_dir.glob("country_codes_V*.csv"))
    if not country_files:
        raise FileNotFoundError("BACI country_codes metadata not found. Run python scripts/download_baci.py")
    df_countries = pd.read_csv(country_files[0])
    
    code_col = [c for c in df_countries.columns if 'code' in c.lower() or c == 'i'][0]
    iso_col = [c for c in df_countries.columns if 'iso' in c.lower() and '3' in c.lower()][0]
    country_map = dict(zip(df_countries[code_col], df_countries[iso_col]))
    
    country_set = set(data_cfg.get("countries", []))
    valid_nums = {num for num, iso in country_map.items() if iso in country_set}
    
    year_start = data_cfg.get("year_start", 2000)
    year_end = data_cfg.get("year_end", 2019)
    
    baci_agg_dfs = []
    
    for y in range(year_start, year_end + 1):
        year_files = list(baci_dir.glob(f"BACI_HS92_Y{y}_V*.csv"))
        if not year_files:
            continue
            
        logger.info("Loading BACI data for year %d from %s", y, year_files[0].name)
        df_y = pd.read_csv(year_files[0], usecols=['t', 'i', 'j', 'k', 'v'])
        
        # Memory Optimization 1: Filter to only the countries in our study
        df_y = df_y[df_y['i'].isin(valid_nums) & df_y['j'].isin(valid_nums)]
        
        if df_y.empty:
            continue
            
        # Map to ISO3 after dropping 90%+ of the rows
        df_y['iso3_o'] = df_y['i'].map(country_map)
        df_y['iso3_d'] = df_y['j'].map(country_map)
        df_y = df_y.dropna(subset=['iso3_o', 'iso3_d'])
        
        # Memory Optimization 2: Aggregate by sector locally per year
        df_y['edge_type'] = df_y['k'].apply(get_hs_section)
        agg_y = df_y.groupby(['t', 'iso3_o', 'iso3_d', 'edge_type'], as_index=False)['v'].sum()
        
        baci_agg_dfs.append(agg_y)
        del df_y
        
    if not baci_agg_dfs:
        logger.warning("No BACI data matched the year range. Proceeding without.")
        df_grav["edge_type"] = 0
        df_grav["log_trade"] = np.log1p(df_grav["tradeflow_comtrade_o"].fillna(0))
        return df_grav
        
    df_baci = pd.concat(baci_agg_dfs, ignore_index=True)
    df_baci = df_baci.rename(columns={'t': 'year'})
    
    logger.info("Merging Gravity with BACI multiplex edges...")
    df_merged = pd.merge(df_baci, df_grav, on=['year', 'iso3_o', 'iso3_d'], how='inner')
    
    df_merged['log_trade'] = np.log1p(df_merged['v'])
    df_merged = df_merged[df_merged['log_trade'] > 0]
    
    return df_merged

def compute_log_trade_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "log_trade" not in df.columns:
        df["log_trade"] = np.log1p(df["tradeflow_comtrade_o"].fillna(0))
    return df

def build_node_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    use_deltas: bool = False,
    num_lags: int = 0,
) -> dict[tuple[str, int], np.ndarray]:
    if feature_cols is None:
        feature_cols = ["gdp", "gdpcap", "pop"]

    node_data: dict[tuple[str, int], np.ndarray] = {}
    
    query_cols = ['year', 'iso3_o', 'iso3_d'] 
    for f in feature_cols:
        query_cols.extend([f"{f}_o", f"{f}_d"])
        for lag in range(1, num_lags + 1):
            query_cols.extend([f"{f}_o_lag{lag}", f"{f}_d_lag{lag}"])
            
    query_cols = [c for c in query_cols if c in df.columns]
    df_nodes = df[query_cols].drop_duplicates()
    
    for _, row in df_nodes.iterrows():
        year = int(row["year"])
        origin = row["iso3_o"]
        dest = row["iso3_d"]

        if (origin, year) not in node_data:
            feats = []
            for f in feature_cols:
                for lag in range(num_lags + 1):
                    col_name = f"{f}_o" if lag == 0 else f"{f}_o_lag{lag}"
                    val = row.get(col_name, np.nan)
                    if use_deltas:
                        feats.append(val if pd.notna(val) else 0.0)
                    else:
                        feats.append(np.log1p(max(0, val)) if pd.notna(val) else 0.0)
            node_data[(origin, year)] = np.array(feats, dtype=np.float32)

        if (dest, year) not in node_data:
            feats = []
            for f in feature_cols:
                for lag in range(num_lags + 1):
                    col_name = f"{f}_d" if lag == 0 else f"{f}_d_lag{lag}"
                    val = row.get(col_name, np.nan)
                    if use_deltas:
                        feats.append(val if pd.notna(val) else 0.0)
                    else:
                        feats.append(np.log1p(max(0, val)) if pd.notna(val) else 0.0)
            node_data[(dest, year)] = np.array(feats, dtype=np.float32)

    return node_data

def build_edge_features(
    row: pd.Series,
    edge_feature_cols: list[str] | None = None,
    num_lags: int = 0,
) -> np.ndarray:
    if edge_feature_cols is None:
        edge_feature_cols = [
            "distw_harmonic", "contig", "comlang_off", "col_dep_ever", "comrelig",
        ]

    feats = []
    for col in edge_feature_cols:
        val = row.get(col, np.nan)
        if pd.isna(val):
            feats.append(0.0)
        elif col == "distw_harmonic":
            feats.append(np.log1p(val))
        else:
            feats.append(float(val))
            
    for lag in range(1, num_lags + 1):
        for col in edge_feature_cols:
            val = row.get(f"{col}_lag{lag}", np.nan)
            if pd.isna(val):
                feats.append(0.0)
            elif col == "distw_harmonic":
                feats.append(np.log1p(max(0, val)) if pd.notna(val) else 0.0)
            else:
                feats.append(float(val))
                
        lag_val = row.get(f"log_trade_lag{lag}", 0.0)
        feats.append(float(lag_val))
            
    return np.array(feats, dtype=np.float32)


def preprocess_pipeline(
    csv_path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = config.get("data", config)
    
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = get_config_hash(data_cfg)
    cache_path = processed_dir / f"trade_data_{config_hash}.parquet"
    
    if cache_path.exists():
        logger.info("Loading preprocessed data from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Cache miss. Running full preprocessing pipeline...")
    
    df = load_and_filter(
        csv_path,
        countries=data_cfg["countries"],
        year_start=data_cfg.get("year_start", 2000),
        year_end=data_cfg.get("year_end", 2019),
    )
    
    if data_cfg.get("use_baci", False):
        df = load_and_merge_baci(df, data_cfg)
    else:
        df["edge_type"] = 0
        df = compute_log_trade_target(df)
        df = df[df["log_trade"] > 0]

    if data_cfg.get("use_deltas", False):
        logger.info("Computing Year-over-Year log-differences (Deltas)...")
        df.sort_values(by=["iso3_o", "iso3_d", "edge_type", "year"], inplace=True)
        
        node_cols = ["gdp_o", "gdp_d", "gdpcap_o", "gdpcap_d", "pop_o", "pop_d"]
        diff_cols = ["log_trade"]
        
        for c in node_cols:
            if c in df.columns:
                df[f"log_{c}"] = np.log1p(df[c].fillna(0).clip(lower=0))
                df[c] = df.groupby(["iso3_o", "iso3_d", "edge_type"])[f"log_{c}"].diff()
                diff_cols.append(c)
                
        if "log_trade" in df.columns:
            df["log_trade"] = df.groupby(["iso3_o", "iso3_d", "edge_type"])["log_trade"].diff()
            
        df = df.dropna(subset=diff_cols)
        df.drop(columns=[f"log_{c}" for c in node_cols if f"log_{c}" in df.columns], inplace=True)
        logger.info(f"Shape after YoY differencing: {df.shape}")

    num_lags = data_cfg.get("num_lags", 0)
    if num_lags > 0:
        logger.info(f"Generating {num_lags} temporal lag features...")
        df.sort_values(by=["iso3_o", "iso3_d", "edge_type", "year"], inplace=True)
        lag_cols = []
        
        cols_to_lag = [c for c in data_cfg.get("node_features", ["gdp", "gdpcap", "pop"]) for suffix in ["_o", "_d"]]
        
        edge_feats = data_cfg.get("edge_features", ["distw_harmonic", "contig", "comlang_off", "col_dep_ever", "comrelig"])
        cols_to_lag.extend(edge_feats)
        
        cols_to_lag = [c for c in cols_to_lag if c in df.columns]
        if "log_trade" in df.columns:
            cols_to_lag.append("log_trade")
            
        group = df.groupby(["iso3_o", "iso3_d", "edge_type"])
        for col in cols_to_lag:
            for lag in range(1, num_lags + 1):
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = group[col].shift(lag)
                lag_cols.append(lag_col)
                
        df = df.dropna(subset=lag_cols)
        logger.info(f"Shape after generating lags: {df.shape}")

    edge_feat_cols = data_cfg.get(
        "edge_features",
        ["distw_harmonic", "contig", "comlang_off", "col_dep_ever", "comrelig"],
    )
    for col in edge_feat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info("Saving preprocessed data to cache: %s", cache_path)
    df.to_parquet(cache_path, index=False)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df
