# Architecture Changes Report

Date: 2026-03-31
Scope: Graph model architecture upgrades to improve competitiveness against tabular baselines (especially XGBoost).

## Summary

We upgraded four graph models with stronger message passing and richer edge decoding patterns:

- `TradeFlowGCN` in `gcn.py`
- `TradeFlowGAT` in `gat.py`
- `TradeFlowEGNN` in `egnn.py`
- `TradeFlowRGCN` in `rgcn.py`

The main design goal was to increase expressive power while improving training stability.

## Files Updated

- `src/trade_flow_gcn/models/gcn.py`
- `src/trade_flow_gcn/models/gat.py`
- `src/trade_flow_gcn/models/egnn.py`
- `src/trade_flow_gcn/models/rgcn.py`

## Common Design Improvements

1. Richer edge-level decoder features

All upgraded models now decode edges using:

- source embedding `h_src`
- destination embedding `h_dst`
- absolute difference `|h_src - h_dst|`
- element-wise product `h_src * h_dst`
- original `edge_attr`

This improves pairwise interaction modeling compared to plain concatenation of `[h_src, h_dst, edge_attr]`.

2. Better optimization stability

Across models we introduced stronger normalization and residual pathways where appropriate to make deeper stacks train more reliably.

## Model-by-Model Changes

### 1) `gcn.py`

Previous:
- Stacked `GCNConv` layers (node-only propagation)
- Standard MLP edge decoder with `[h_src, h_dst, edge_attr]`

Now:
- Replaced with edge-aware `GINEConv` blocks
- Added residual connections and `LayerNorm` in each block
- Upgraded decoder to richer interaction feature set

Expected impact:
- Better use of edge covariates during node encoding
- More expressive link/flow prediction at decode stage

### 2) `gat.py`

Previous:
- `GATConv` stack with attention over node structure
- Decoder with basic concatenation

Now:
- Switched to edge-aware `GATv2Conv` (`edge_dim` used)
- Added residual + `LayerNorm` in custom block
- Upgraded decoder to richer interaction feature set

Expected impact:
- Attention scores can incorporate edge attributes
- Better representational power for directed trade edges

### 3) `egnn.py`

Previous:
- Edge-conditioned message function
- Node update without explicit residual normalization

Now:
- Added edge-conditioned message gate (`sigmoid`) to modulate message strength
- Added residual projection + `LayerNorm` in update step
- Upgraded decoder to richer interaction feature set

Expected impact:
- Cleaner control of noisy messages
- Improved gradient flow and layer stability

### 4) `rgcn.py`

Previous:
- `RGCNConv` stack without explicit residual/normalization
- Distance binning thresholds compared directly against transformed distance

Now:
- Added residual + `LayerNorm` in each RGCN block
- Corrected relation thresholding to log-space (`log1p(5000)`, `log1p(10000)`) to align with upstream transformed distance feature
- Upgraded decoder to richer interaction feature set

Expected impact:
- More consistent relation typing with preprocessing
- Better training stability in multi-layer relational encoder

## Validation Performed

- Python compile check on all modified model files: passed
- Smoke tests (`tests/test_smoke.py`): passed (13/13)

## Git Status Note

At report creation time, these architecture changes are local workspace changes and have not been committed/pushed yet.

## Suggested Next Evaluation Step

1. Retrain updated graph models with identical data split and seed.
2. Re-run unified benchmark against tabular baselines.
3. Compare RMSE/MAE/R2 and training stability trends across runs.
