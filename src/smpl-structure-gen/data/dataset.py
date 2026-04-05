"""PyTorch Geometric dataset and Lightning DataModule for trade graphs.

Each year produces one ``torch_geometric.data.Data`` object with:
- ``x``: node features ``(N, F_node)``
- ``edge_index``: directed edges ``(2, E)``
- ``edge_attr``: edge features ``(E, F_edge)``
- ``edge_type``: corresponding to BACI product category ``(E, )``
- ``y``: log-trade targets ``(E,)``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from trade_flow_gcn.data.preprocessing import (
    build_edge_features,
    build_node_features,
)

logger = logging.getLogger(__name__)


def build_graph_for_year(
    df_year: pd.DataFrame,
    country_list: list[str],
    node_feature_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    use_baci: bool = False,
    use_deltas: bool = False,
    num_lags: int = 0,
) -> Data:
    country_to_idx = {c: i for i, c in enumerate(country_list)}
    year = int(df_year["year"].iloc[0])

    node_data = build_node_features(df_year, feature_cols=node_feature_cols, use_deltas=use_deltas, num_lags=num_lags)
    n_nodes = len(country_list)
    n_node_feat = len(node_feature_cols or ["gdp", "gdpcap", "pop"]) * (num_lags + 1)
    x = torch.zeros(n_nodes, n_node_feat, dtype=torch.float32)
    for country in country_list:
        idx = country_to_idx[country]
        key = (country, year)
        if key in node_data:
            x[idx] = torch.from_numpy(node_data[key])

    src_indices = []
    dst_indices = []
    edge_feats = []
    edge_types = []
    
    num_sectors = 21 if use_baci else 1
    y_nodes = torch.zeros(n_nodes, num_sectors, dtype=torch.float32)

    for _, row in df_year.iterrows():
        o = row["iso3_o"]
        d = row["iso3_d"]
        if o not in country_to_idx or d not in country_to_idx:
            continue
            
        o_idx = country_to_idx[o]
        d_idx = country_to_idx[d]
        
        src_indices.append(o_idx)
        dst_indices.append(d_idx)
        
        base_edge_feat = build_edge_features(row, edge_feature_cols, num_lags=num_lags)
        
        e_type = int(row.get("edge_type", 0)) if use_baci else 0
        
        if use_baci:
            one_hot = np.zeros(21, dtype=np.float32)
            if 0 <= e_type <= 20:
                one_hot[e_type] = 1.0
            final_edge_feat = np.concatenate([base_edge_feat, one_hot])
            edge_types.append(e_type)
        else:
            final_edge_feat = base_edge_feat
            edge_types.append(0)
        
        edge_feats.append(final_edge_feat)
        
        # PIVOT A: Node-Level Target (Aggregate Outgoing Export Deltas per Sector)
        # Represents Sector Vulnerability/Growth Index for the exporting country
        safe_e_type = min(e_type, num_sectors - 1)
        y_nodes[o_idx, safe_e_type] += row["log_trade"]

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    base_edge_dim = len(edge_feature_cols or [])
    lagged_edge_dim = base_edge_dim * (num_lags + 1) + num_lags
    edge_attr = torch.from_numpy(np.stack(edge_feats)).float() if edge_feats else torch.empty((0, lagged_edge_dim + (21 if use_baci else 0)))
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # ASSIGN PIVOT A TARGET TO BATCH.Y
    y = y_nodes

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_type = edge_type
    data.year = year
    data.country_list = country_list

    return data


def build_graphs_from_dataframe(
    df: pd.DataFrame,
    country_list: list[str],
    config: dict[str, Any] | None = None,
) -> list[Data]:
    data_cfg = (config or {}).get("data", config or {})
    use_baci = data_cfg.get("use_baci", False)
    use_deltas = data_cfg.get("use_deltas", False)
    num_lags = data_cfg.get("num_lags", 0)
    node_feat_cols = data_cfg.get("node_features", ["gdp", "gdpcap", "pop"])
    edge_feat_cols = data_cfg.get(
        "edge_features",
        ["distw_harmonic", "contig", "comlang_off", "col_dep_ever", "comrelig"],
    )

    graphs = []
    for year, df_year in sorted(df.groupby("year")):
        logger.info("Building graph for year %d (%d edges) ...", year, len(df_year))
        g = build_graph_for_year(
            df_year,
            country_list=country_list,
            node_feature_cols=node_feat_cols,
            edge_feature_cols=edge_feat_cols,
            use_baci=use_baci,
            use_deltas=use_deltas,
            num_lags=num_lags,
        )
        graphs.append(g)

    return graphs


class TradeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        graphs: list[Data],
        train_years: tuple[int, int] = (2000, 2014),
        val_years: tuple[int, int] = (2015, 2016),
        test_years: tuple[int, int] = (2017, 2019),
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.graphs = graphs
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.num_workers = num_workers

        self.train_graphs: list[Data] = []
        self.val_graphs: list[Data] = []
        self.test_graphs: list[Data] = []

    def setup(self, stage: str | None = None) -> None:
        self.train_graphs = []
        self.val_graphs = []
        self.test_graphs = []

        for g in self.graphs:
            y = g.year
            if self.train_years[0] <= y <= self.train_years[1]:
                self.train_graphs.append(g)
            elif self.val_years[0] <= y <= self.val_years[1]:
                self.val_graphs.append(g)
            elif self.test_years[0] <= y <= self.test_years[1]:
                self.test_graphs.append(g)

        logger.info(
            "Data split — train: %d graphs, val: %d graphs, test: %d graphs",
            len(self.train_graphs),
            len(self.val_graphs),
            len(self.test_graphs),
        )

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_graphs, 
            batch_size=1, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_graphs, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_graphs, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers
        )
