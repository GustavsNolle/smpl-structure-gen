"""Lightning module for Causal Graph Information Bottleneck.

Isolates a minimal Pharmacophore (Causal Subgraph) while forcing the Scaffold
(Environment Subgraph) to discard spurious correlations.
"""

import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanSquaredError
from torchmetrics.classification import BinaryAUROC
from clearml import Task

logger = logging.getLogger(__name__)


class CausalSemiSupModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        task_types: list[str],
        dataset_names: list[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        sparsity_beta: float = 1.0,
        env_beta: float = 0.5,
        target_to_ds: dict[str, str] | None = None,
        model_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.task_types = task_types
        self.dataset_names = dataset_names
        self.target_to_ds = target_to_ds or {name: "default" for name in dataset_names}
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sparsity_beta = sparsity_beta
        self.env_beta = env_beta

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.train_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        self.val_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        self.test_auroc = nn.ModuleList([BinaryAUROC() if tt == "classification" else None for tt in task_types])
        
        self.train_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        self.val_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        self.test_rmse = nn.ModuleList([MeanSquaredError(squared=False) if tt == "regression" else None for tt in task_types])
        
        self.latest_test_results = {}

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

    def _shared_step(self, batch, stage: str):
        pred_c, pred_e, mask = self(batch)
        y = batch.y.view(-1, len(self.task_types))

        causal_loss = 0.0
        env_loss = 0.0

        for i, (tt, name) in enumerate(zip(self.task_types, self.dataset_names)):
            task_pred_c = pred_c[:, i]
            task_pred_e = pred_e[:, i]
            task_target = y[:, i]
            
            mask_valid = ~torch.isnan(task_target)
            if not mask_valid.any():
                continue
            
            valid_pc = task_pred_c[mask_valid]
            valid_pe = task_pred_e[mask_valid]
            valid_target = task_target[mask_valid]
            
            if tt == "classification":
                lc = self.bce_loss(valid_pc, valid_target)
                le = self.bce_loss(valid_pe, valid_target)
                
                causal_loss += lc
                env_loss += le
                
                # Update metrics on Causal Subgraph ONLY (this is our true predictor)
                if stage == "train":
                    self.train_auroc[i](valid_pc, valid_target.long())
                elif stage == "val":
                    self.val_auroc[i](valid_pc, valid_target.long())
                elif stage == "test":
                    self.test_auroc[i](valid_pc, valid_target.long())
                    
            elif tt == "regression":
                lc = self.mse_loss(valid_pc, valid_target)
                le = self.mse_loss(valid_pe, valid_target)
                
                causal_loss += lc * 0.1
                env_loss += le * 0.1
                
                if stage == "train":
                    self.train_rmse[i](valid_pc, valid_target)
                elif stage == "val":
                    self.val_rmse[i](valid_pc, valid_target)
                elif stage == "test":
                    self.test_rmse[i](valid_pc, valid_target)

        # Objective Function
        total_loss = causal_loss
        
        # 1. Sparsity Loss (Information Bottleneck)
        # Forces the mask to be as sparse as possible (close to 0), squeezing out everything
        # except the most causal variables necessary to satisfy `causal_loss`.
        mask_loss = mask.mean()
        if self.sparsity_beta > 0:
            total_loss += self.sparsity_beta * mask_loss
            
        # 2. Environment Entropy Penalty (IRM-lite)
        # If the environment loss is SMALL, it means the environment contains spurious features 
        # perfectly matching the label. We penalize this heavily!
        # exp(-env_loss) approaches 1 when env_loss is 0, and 0 when env_loss is large.
        env_penalty = torch.tensor(0.0, device=self.device)
        if self.env_beta > 0 and hasattr(env_loss, 'item') and env_loss > 0:
            env_penalty = self.env_beta * torch.exp(-env_loss)
            total_loss += env_penalty

        # Detach for logging
        with torch.no_grad():
            detached_loss = total_loss.detach().cpu().item()
            self.log(f"{stage}/loss", detached_loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_loss", detached_loss, batch_size=batch.num_graphs, sync_dist=True)
            self.log(f"{stage}/mask_sparsity", mask_loss.detach().cpu(), batch_size=batch.num_graphs, sync_dist=True)
            if self.env_beta > 0 and hasattr(env_loss, 'item'):
                self.log(f"{stage}/env_penalty", env_penalty.detach().cpu(), batch_size=batch.num_graphs, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        stage = "val" if dataloader_idx == 0 else "test"
        return self._shared_step(batch, stage)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")
        self._log_epoch_metrics("test")

    def _log_epoch_metrics(self, stage: str) -> dict[str, float]:
        overall_metric = 0.0
        num_valid = 0
        results = {}
        
        task = Task.current_task()
        cl_logger = task.get_logger() if task else None
        
        for i, (tt, name) in enumerate(zip(self.task_types, self.dataset_names)):
            if tt == "classification":
                metric_obj = getattr(self, f"{stage}_auroc")[i]
                try:
                    val = metric_obj.compute().item()
                    ds_name = self.target_to_ds.get(name, "unknown").upper()
                    
                    self.log(f"{stage}/{ds_name} C-AUROC/{name}", val, prog_bar=(stage != "train"), sync_dist=True)
                    if cl_logger and stage == "test":
                        cl_logger.report_single_value(name=f"TEST_{ds_name}_C-AUROC_{name}", value=val)
                        
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
                    
                    self.log(f"{stage}/{ds_name} C-RMSE/{name}", val, prog_bar=(stage != "train"), sync_dist=True)
                    if cl_logger and stage == "test":
                        cl_logger.report_single_value(name=f"TEST_{ds_name}_C-RMSE_{name}", value=val)
                        
                    results[f"{stage}_{name}_rmse"] = val
                    overall_metric += -val 
                    num_valid += 1
                except ValueError:
                    pass
                metric_obj.reset()

        if num_valid > 0:
            avg_score = overall_metric / num_valid
            self.log(f"{stage}/overall_c_score", avg_score, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_overall_c_score", avg_score, sync_dist=True)
            # Route target to val_loss for checkpoint callback matching
            if stage == "val":
                self.log(f"val_loss", -avg_score, prog_bar=False, sync_dist=True)
                
            results[f"{stage}_overall_score"] = avg_score
            
        return results

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_test_epoch_end(self):
        results = self._log_epoch_metrics("test")
        self.latest_test_results = results
        
        logger.info("=========================================")
        logger.info(" CAUSAL TEST REPORT ")
        logger.info("=========================================")
        for name, val in results.items():
            if "overall" not in name:
                logger.info(" => %s: %.4f", name.replace("test_", "").upper(), val)
        logger.info("-----------------------------------------")
        logger.info(" => CAUSAL OVERALL SCORE: %.4f", results.get("test_overall_score", 0.0))
        logger.info("=========================================")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
