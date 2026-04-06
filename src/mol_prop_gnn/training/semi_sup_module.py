"""Lightning module for Semi-Supervised Joint Embedding.

Handles multi-dataset NaN-masked training and dimensionality constraint losses.
"""

import logging
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanSquaredError
from torchmetrics.classification import BinaryAUROC
from clearml import Task

from mol_prop_gnn.data.augmentations import GraphAugmentor, augment_batch

logger = logging.getLogger(__name__)


def masked_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    """Compute loss only on valid (non-NaN) targets."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss_fn(pred[mask], target[mask])


class JointSemiSupModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        task_types: list[str],
        dataset_names: list[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        ortho_beta: float = 0.01,
        contrastive_beta: float = 0.1,
        temperature: float = 0.07,
        target_to_ds: dict[str, str] | None = None,
        model_config: dict | None = None,
    ) -> None:
        super().__init__()
        # Uncertainty Weighting uses standard automatic optimization
        self.automatic_optimization = True
        self.model = model
        self.task_types = task_types
        self.dataset_names = dataset_names
        self.target_to_ds = target_to_ds or {name: "default" for name in dataset_names}
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ortho_beta = ortho_beta
        self.contrastive_beta = contrastive_beta
        self.temperature = temperature

        self.augmentor = GraphAugmentor(node_drop_p=0.05, edge_mask_p=0.05)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

        # Metrics lists
        self.train_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        self.val_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        self.test_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        
        self.train_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        self.val_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        self.test_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        
        # Learnable noise parameters for Uncertainty Weighting
        # Initialized at 0.0, which corresponds to weighing each loss by exp(-0) = 1.0
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_types)))

        # Persistent storage for final reporting to avoid Lightning's manual-optimization return bug
        self.latest_test_results = {}
        self.validation_step_outputs = []

    def forward(self, batch) -> torch.Tensor:
        return self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

    def _compute_disentanglement_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Enforces that the N semantic dimensions encode completely orthogonal information.
        Penalizes correlation between columns in a batch.
        """
        if h.size(0) < 2:
            return torch.tensor(0.0, device=h.device)
            
        # Center the batch representations
        h_centered = h - h.mean(dim=0, keepdim=True)
        # Covariance matrix
        cov = (h_centered.T @ h_centered) / (h.size(0) - 1)
        
        # We want off-diagonals to be 0
        diag_mask = torch.eye(cov.size(0), device=cov.device, dtype=torch.bool)
        off_diag = cov[~diag_mask]
        
        return torch.norm(off_diag, p=2)

    def _compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent (InfoNCE) loss for contrastive learning between two Views."""
        # Normalize to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        batch_size = z1.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device)
            
        # Concatenate both views [2N, D]
        out = torch.cat([z1, z2], dim=0)
        
        # Similarity matrix [2N, 2N]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        
        # Mask out self-similarities
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=out.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        
        # Positive pairs (diagonal of z1*z2 block)
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # Match the 2N dimension
        
        # Loss: -log(pos / sum(neg))
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def _shared_step(self, batch, stage: str, log: bool = True):
        # Forward pass yields (B, num_datasets) footprint
        pred = self(batch)
        y = batch.y.view(-1, len(self.task_types))

        total_loss = 0.0
        losses = []

        # 1. Prediction Losses per dimension
        for i, (tt, name) in enumerate(zip(self.task_types, self.dataset_names)):
            task_pred = pred[:, i]
            task_target = y[:, i]
            
            mask = ~torch.isnan(task_target)
            if not mask.any():
                continue
            
            valid_pred = task_pred[mask]
            valid_target = task_target[mask]
            
            if tt == "classification":
                l = self.bce_loss(valid_pred, valid_target)
                total_loss += l
                losses.append(l.item())
                # Update metrics
                if stage == "train":
                    self.train_auroc[i](valid_pred, valid_target.long())
                elif stage == "val":
                    self.val_auroc[i](valid_pred, valid_target.long())
                elif stage == "test":
                    self.test_auroc[i](valid_pred, valid_target.long())
                    
            elif tt == "regression":
                l = self.mse_loss(valid_pred, valid_target)
                # Downweight regression loss so it doesn't overpower BCE
                total_loss += l * 0.1
                losses.append(l.item())
                if stage == "train":
                    self.train_rmse[i](valid_pred, valid_target)
                elif stage == "val":
                    self.val_rmse[i](valid_pred, valid_target)
                elif stage == "test":
                    self.test_rmse[i](valid_pred, valid_target)

        # 2. Constraint / Disentanglement Loss
        d_loss = self._compute_disentanglement_loss(pred)
        total_loss += self.ortho_beta * d_loss
        
        if log:
            # .detach().cpu() is the critical fix for Manual Optimization
            # It prevents memory leaks and ensures ModelCheckpoint can 'read' the value
            detached_loss = total_loss.detach().cpu().item()
            
            # We log both f"{stage}/loss" for plotting and f"{stage}_loss" for ModelCheckpoint monitoring
            self.log(f"{stage}/loss", detached_loss, batch_size=batch.num_graphs, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_loss", detached_loss, batch_size=batch.num_graphs, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        """Standard training step with automatic Uncertainty Weighting."""
        pred = self(batch)
        y = batch.y.view(-1, len(self.task_types))
        
        total_loss = 0.0
        
        # 1. Supervised Weighted Losses
        for i, (tt, name) in enumerate(zip(self.task_types, self.dataset_names)):
            task_pred = pred[:, i]
            task_target = y[:, i]
            mask = ~torch.isnan(task_target)
            if not mask.any():
                continue
            
            valid_pred = task_pred[mask]
            valid_target = task_target[mask]
            
            if tt == "classification":
                l = self.bce_loss(valid_pred, valid_target)
                self.train_auroc[i](valid_pred, valid_target.long())
            elif tt == "regression":
                l = self.mse_loss(valid_pred, valid_target)
                self.train_rmse[i](valid_pred, valid_target)
                
            # Uncertainty Weighting: (l * exp(-log_var)) + log_var
            precision = torch.exp(-self.log_vars[i])
            total_loss += (l * precision) + self.log_vars[i]

        # 2. Auxiliary Losses
        # Global losses remain fixed with their respective betas
        d_loss = self._compute_disentanglement_loss(pred)
        if self.ortho_beta > 0:
            total_loss += self.ortho_beta * d_loss
            
        c_loss = torch.tensor(0.0, device=self.device)
        if self.contrastive_beta > 0:
            aug_batch = augment_batch(batch, self.augmentor).to(self.device)
            pred_aug = self(aug_batch)
            c_loss = self._compute_contrastive_loss(pred, pred_aug)
            total_loss += self.contrastive_beta * c_loss
            
        # 3. Logging
        # Detach for logging to avoid tracking graphs
        with torch.no_grad():
            detached_loss = total_loss.detach().cpu().item()
            self.log("train/loss", detached_loss, batch_size=batch.num_graphs, prog_bar=True)
            self.log("train_loss", detached_loss, batch_size=batch.num_graphs)
            self.log("train/disentangle_loss", d_loss.detach().cpu(), batch_size=batch.num_graphs)
            self.log("train/contrastive_loss", c_loss.detach().cpu(), batch_size=batch.num_graphs)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        """Stable reporting of val_loss and epoch metrics."""
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, stage: str) -> dict[str, float]:
        overall_metric = 0.0
        num_valid = 0
        results = {}
        
        # Access ClearML task for direct reporting
        task = Task.current_task()
        cl_logger = task.get_logger() if task else None
        
        for i, (tt, name) in enumerate(zip(self.task_types, self.dataset_names)):
            if tt == "classification":
                metric_obj = getattr(self, f"{stage}_auroc")[i]
                try:
                    val = metric_obj.compute().item()
                    ds_name = self.target_to_ds.get(name, "unknown").upper()
                    # Grouping: stage / [Dataset] [MetricType] / [Target]
                    # ClearML will create a plot named "[Dataset] AUROC" with series "[Target]"
                    log_key = f"{stage}/{ds_name} AUROC/{name}"
                    self.log(log_key, val, prog_bar=(stage != "train"), on_epoch=True, sync_dist=True)
                    
                    if cl_logger:
                        cl_logger.report_scalar(title=f"{stage.upper()} {ds_name} AUROC", series=name, value=val, iteration=self.current_epoch)
                    
                    results[f"{stage}_{name}_auroc"] = val
                    overall_metric += val
                    num_valid += 1
                except ValueError:
                    pass
                metric_obj.reset()
            else:
                metric_obj = getattr(self, f"{stage}_rmse")[i]
                try:
                    val = metric_obj.compute().item()
                    ds_name = self.target_to_ds.get(name, "unknown").upper()
                    # Grouping: stage / [Dataset] [MetricType] / [Target]
                    log_key = f"{stage}/{ds_name} RMSE/{name}"
                    self.log(log_key, val, prog_bar=(stage != "train"), on_epoch=True, sync_dist=True)
                    
                    if cl_logger:
                        cl_logger.report_scalar(title=f"{stage.upper()} {ds_name} RMSE", series=name, value=val, iteration=self.current_epoch)
                    
                    results[f"{stage}_{name}_rmse"] = val
                    # We negate RMSE so that "higher is better" for model checkpointing
                    overall_metric += -val 
                    num_valid += 1
                except ValueError:
                    pass
                metric_obj.reset()

        if num_valid > 0:
            avg_score = overall_metric / num_valid
            self.log(f"{stage}/overall_score", avg_score, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_overall_score", avg_score, on_epoch=True, sync_dist=True)
            results[f"{stage}_overall_score"] = avg_score
            
        return results

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_test_epoch_end(self):
        """Final reporting for the test phase."""
        results = self._log_epoch_metrics("test")
        self.latest_test_results = results
        
        # Log summary to console
        logger.info("=========================================")
        logger.info(" FINAL TEST REPORT ")
        logger.info("=========================================")
        for name, val in results.items():
            if "overall" not in name:
                logger.info(" => %s: %.4f", name.replace("test_", "").upper(), val)
        logger.info("-----------------------------------------")
        logger.info(" => OVERALL TEST SCORE: %.4f", results.get("test_overall_score", 0.0))
        logger.info("=========================================")

    def configure_optimizers(self):
        # Optimized model + the log_vars weight parameters are all captured by self.parameters()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
