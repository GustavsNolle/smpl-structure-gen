"""Run model benchmark and save leaderboard CSV without changing existing code.

Usage:
    python scripts/experiments/benchmark_models_to_csv.py
    python scripts/experiments/benchmark_models_to_csv.py --output results/model_comparison.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch_geometric

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trade_flow_gcn.data.dataset import TradeDataModule, build_graphs_from_dataframe
from trade_flow_gcn.data.preprocessing import preprocess_pipeline

from trade_flow_gcn.models.gcn import TradeFlowGCN
from trade_flow_gcn.models.gravity_baseline import GravityBaseline
from trade_flow_gcn.models.hybrid_gae_xgboost import HybridGAEXGBoost
from trade_flow_gcn.models.lightgbm_baseline import LightGBMBaseline
from trade_flow_gcn.models.mlp_baseline import MLPBaseline
from trade_flow_gcn.models.rgcn import TradeFlowRGCN
from trade_flow_gcn.models.xgboost_baseline import XGBoostBaseline
from trade_flow_gcn.training.lightning_module import TradeFlowModule
from trade_flow_gcn.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_numpy(data_list: List):
    x_src, x_dst, edge_attr, y = [], [], [], []
    for graph in data_list:
        src, dst = graph.edge_index
        x_src.append(graph.x[src].numpy())
        x_dst.append(graph.x[dst].numpy())
        edge_attr.append(graph.edge_attr.numpy())
        y.append(graph.y.numpy())
    return (
        np.concatenate(x_src),
        np.concatenate(x_dst),
        np.concatenate(edge_attr),
        np.concatenate(y),
    )


def evaluate_torch_model(module: TradeFlowModule, graphs) -> dict[str, float]:
    device = next(module.parameters()).device
    loader = torch_geometric.loader.DataLoader(graphs, batch_size=1)
    preds, targets = [], []
    module.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = module(batch)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())

    preds_np = np.concatenate(preds)
    targets_np = np.concatenate(targets)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets_np, preds_np))),
        "mae": float(mean_absolute_error(targets_np, preds_np)),
        "r2": float(r2_score(targets_np, preds_np)),
    }


def load_dl_model(model_type: str, model_class, base_dir: Path, config: dict):
    ckpt_files = list(base_dir.glob(f"{model_type}/**/checkpoints/*.ckpt"))
    if not ckpt_files:
        ckpt_files = [p for p in base_dir.glob("**/checkpoints/*.ckpt") if model_type in str(p)]

    if not ckpt_files:
        logger.warning("No checkpoint found for %s", model_type)
        return None

    latest_ckpt = sorted(ckpt_files, key=lambda p: p.stat().st_mtime)[-1]

    if model_type in {"gcn", "rgcn"}:
        net = model_class(
            node_input_dim=len(config["data"]["node_features"]),
            edge_input_dim=len(config["data"]["edge_features"]),
        )
    else:
        net = model_class(
            input_dim=len(config["data"]["node_features"]) * 2 + len(config["data"]["edge_features"]),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TradeFlowModule.load_from_checkpoint(latest_ckpt, model=net).to(device)
    logger.info("Loaded %s from %s", model_type, latest_ckpt.name)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all available models and save CSV")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--output", type=str, default="results/model_comparison.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(".")
    log_dir = root / args.log_dir

    raw_dir = root / config["data"]["raw_dir"]
    csv_candidates = list(raw_dir.glob("*.csv"))
    if not csv_candidates:
        logger.error("No CSV file found in %s", raw_dir)
        return 1

    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config["data"]["countries"], config)

    dm = TradeDataModule(
        graphs=graphs,
        train_years=tuple(config["data"]["train_years"]),
        val_years=tuple(config["data"]["val_years"]),
        test_years=tuple(config["data"]["test_years"]),
    )
    dm.setup()

    results: list[dict[str, float | str]] = []

    logger.info("Evaluating tabular baselines...")
    x_s_train, x_d_train, e_train, y_train = extract_numpy(dm.train_graphs)
    x_s_val, x_d_val, e_val, y_val = extract_numpy(dm.val_graphs)
    x_s_test, x_d_test, e_test, y_test = extract_numpy(dm.test_graphs)

    gravity = GravityBaseline()
    gravity.fit(x_s_train, x_d_train, e_train, y_train)
    results.append({"Model": "Gravity (OLS)", **gravity.evaluate(x_s_test, x_d_test, e_test, y_test)})

    xgb = XGBoostBaseline()
    xgb_val_x = np.concatenate([x_s_val, x_d_val, e_val], axis=1)
    xgb.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(xgb_val_x, y_val)])
    results.append({"Model": "XGBoost", **xgb.evaluate(x_s_test, x_d_test, e_test, y_test)})

    lgb_model = LightGBMBaseline()
    lgb_model.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(xgb_val_x, y_val)])
    results.append({"Model": "LightGBM", **lgb_model.evaluate(x_s_test, x_d_test, e_test, y_test)})

    logger.info("Evaluating deep learning checkpoints...")
    models_to_test = [
        ("gcn", TradeFlowGCN),

        ("rgcn", TradeFlowRGCN),
        ("mlp_baseline", MLPBaseline),
    ]

    for model_name, model_class in models_to_test:
        module = load_dl_model(model_name, model_class, log_dir, config)
        if module is None:
            continue
        pretty_name = model_name.replace("_", " ").upper()
        results.append({"Model": f"TradeFlow {pretty_name}", **evaluate_torch_model(module, dm.test_graphs)})

    embedding_path = root / "data" / "processed" / "node_embeddings.npy"
    if embedding_path.exists():
        logger.info("Evaluating Hybrid (GAE + XGBoost)...")
        embeddings_dict = np.load(embedding_path, allow_pickle=True).item()
        hybrid_model = HybridGAEXGBoost()
        hybrid_model.set_embeddings(embeddings_dict)
        try:
            hybrid_model.fit(
                dm.train_graphs,
                dm.val_graphs,
                train_start_idx=0,
                val_start_idx=len(dm.train_graphs),
            )
            h_metrics = hybrid_model.evaluate(
                dm.test_graphs,
                start_idx=len(dm.train_graphs) + len(dm.val_graphs),
            )
            results.append({"Model": "Hybrid (GAE + XGBoost)", **h_metrics})
        except Exception as exc:
            logger.error("Hybrid model failed: %s", exc)

    summary_df = pd.DataFrame(results).sort_values("rmse")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON RESULTS")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("=" * 72)
    print(f"Saved CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
