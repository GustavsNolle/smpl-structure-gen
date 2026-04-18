import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanSquaredError
from torchmetrics.classification import BinaryAUROC

logger = logging.getLogger(__name__)

class MolPropertyModule(pl.LightningModule):
    """Simple supervised Lightning module for a single task."""
    
    def __init__(self, model, task_type="classification", learning_rate=1e-3, weight_decay=1e-4, scheduler_config=None):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, R2Score
        
        if task_type == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.train_metric = BinaryAUROC()
            self.val_metric = BinaryAUROC()
            self.test_metric = BinaryAUROC()
            # Also track accuracy
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
        else:
            self.loss_fn = nn.MSELoss()
            self.train_metric = MeanSquaredError(squared=False) # RMSE
            self.val_metric = MeanSquaredError(squared=False)
            self.test_metric = MeanSquaredError(squared=False)
            
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()
            
            self.val_r2 = R2Score()
            self.test_r2 = R2Score()
            
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        if hasattr(batch, 'edge_index'):
            return self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, 'edge_attr', None),
                batch=batch.batch,
            )
        else:
            # Fallback for tabular features if ever used directly in this module
            return self.model(batch)

    def _shared_step(self, batch, batch_idx, metric, extra_metrics, prefix):
        pred = self(batch)
        y = batch.y.view(-1, 1).float()
        
        # Valid mask
        valid = ~torch.isnan(y)
        if not valid.any():
            return None
        
        pred_v = pred[valid]
        y_v = y[valid]
        
        loss = self.loss_fn(pred_v, y_v)
        
        # BinaryAUROC expects probability / logits in 1D, y in 1D long/int
        if self.task_type == "classification":
            metric(pred_v.view(-1), y_v.view(-1).long())
            if extra_metrics is not None:
                extra_metrics(pred_v.view(-1), y_v.view(-1).long())
                self.log(f"{prefix}_acc", extra_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=valid.sum())
        else:
            metric(pred_v, y_v)
            if extra_metrics is not None:
                mae_metric, r2_metric = extra_metrics
                mae_metric(pred_v, y_v)
                if pred_v.size(0) > 1:
                    r2_metric(pred_v, y_v)
                else:
                    r2_metric.update(pred_v, y_v)
                
                self.log(f"{prefix}_mae", mae_metric, on_step=False, on_epoch=True, prog_bar=False, batch_size=valid.sum())
                self.log(f"{prefix}_r2", r2_metric, on_step=False, on_epoch=True, prog_bar=False, batch_size=valid.sum())
            
        self.log(f"{prefix}_loss", loss, batch_size=valid.sum())
        self.log(f"{prefix}_{'auroc' if self.task_type == 'classification' else 'rmse'}", metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=valid.sum())
        
        return loss

    def training_step(self, batch, batch_idx):
        if self.task_type == "classification":
            extra = getattr(self, "train_acc", None)
        else:
            extra = None  # We didn't initialize train_mae / train_r2
        return self._shared_step(batch, batch_idx, self.train_metric, extra, "train")

    def validation_step(self, batch, batch_idx):
        if self.task_type == "classification":
            extra = getattr(self, "val_acc", None)
        else:
            extra = (getattr(self, "val_mae"), getattr(self, "val_r2"))
        return self._shared_step(batch, batch_idx, self.val_metric, extra, "val")

    def test_step(self, batch, batch_idx):
        if self.task_type == "classification":
            extra = getattr(self, "test_acc", None)
        else:
            extra = (getattr(self, "test_mae"), getattr(self, "test_r2"))
        return self._shared_step(batch, batch_idx, self.test_metric, extra, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        if self.scheduler_config:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_config)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer
        
