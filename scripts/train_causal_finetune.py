"""Causal Fine-Tuning Script.

Pulls a pre-trained GINE model from ClearML, strips the pre-training prediction head,
and fine-tunes it using a hybrid Causal + Contrastive + Uncertainty-Weighted objective
across a filtered set of MoleculeNet datasets (excluding HIV).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from clearml import Task, Model as ClearMLModel

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, BaseFinetuning, StochasticWeightAveraging

from mol_prop_gnn.data.unified_dataset import build_unified_dataframe, preprocess_unified_dataset
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.gine_sixseeven import MolGINE
from mol_prop_gnn.models.hybrid_casual import CausalContrastiveUncertaintyEmbedder
from mol_prop_gnn.training.causal_semi_sup_module import CausalSemiSupModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


class BackboneWarmupCallback(BaseFinetuning):
    def __init__(self, unfreeze_epoch: int = 10, lr_multiplier: float = 0.1):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_multiplier = lr_multiplier

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        # Freeze the entire backbone (MolGINE)
        self.freeze(pl_module.model.backbone)
        logger.info("Backbone frozen for initial warm-up phase.")
        
    def finetune_function(self, pl_module: pl.LightningModule, current_epoch: int, optimizer: torch.optim.Optimizer) -> None:
        if current_epoch == self.unfreeze_epoch:
            # Unfreeze backbone and add it to optimizer with a lower learning rate
            base_lr = pl_module.learning_rate
            backbone_lr = base_lr * self.lr_multiplier
            
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.backbone,
                optimizer=optimizer,
                lr=backbone_lr,
                train_bn=True,
            )
            
            # CRITICAL FIX: Synchronize the PyTorch scheduler to prevent it from crashing
            # because the optimizer now has 3 parameter groups instead of 2.
            for config in pl_module.trainer.lr_scheduler_configs:
                scheduler = config.scheduler
                if hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs.append(backbone_lr)
                    
            logger.info(f"Epoch {current_epoch}: Unfroze Backbone. Backbone LR set to {backbone_lr:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Fine-Tuning of Pre-trained GINE")
    parser.add_argument("--clearml_id", type=str, default="d4af025ccb194adfac886cba6f2026b8",
                        help="ClearML ID of the pre-trained GINE checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GINE backbone")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GINE message-passing layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Semantic bottleneck dimension")
    parser.add_argument("--sparsity_beta", type=float, default=0.1, help="Sparsity constraint weight")
    parser.add_argument("--env_beta", type=float, default=0.4, help="Environment penalty weight")
    parser.add_argument("--contrastive_temp", type=float, default=0.07, help="Contrastive loss temperature")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--split_type", type=str, default="stratified_butina", help="Data splitting methodology")
    parser.add_argument("--similarity_cutoff", type=float, default=0.4, help="Similarity cutoff for Butina clustering")
    parser.add_argument("--accelerator", type=str, default="auto", help="Hardware accelerator (auto, cpu, gpu)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--unfreeze_epoch", type=int, default=10, help="Epoch to unfreeze the pre-trained backbone")
    # We exclude HIV as requested
    parser.add_argument("--datasets", nargs="+", default=["bbbp", "esol", "freesolv", "lipophilicity", "bace", "tox21", "clintox", "herg", "sascore"], 
                        help="Datasets to fine-tune on")
    
    args = parser.parse_args()
    
    # Initialize ClearML tracking
    task = Task.init(
        project_name="MoleculeNet-Causal", 
        task_name=f"gine_causal_finetune_{args.clearml_id[:6]}",
        output_uri=True
    )
    task.connect(args)
    
    logger.info("Initializing multi-dataset assembly. Datasets: %s", args.datasets)
    df, scaling_stats, target_names, task_types, target_to_ds = build_unified_dataframe(dataset_names=args.datasets)
    
    graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
        df, 
        target_names=target_names,
        split_type=args.split_type,
        similarity_cutoff=args.similarity_cutoff
    )
    
    # We use the balanced sampler naturally as requested by the original causal pipeline
    datamodule = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampler=True
    )
    datamodule.setup()
    
    # 2. Retrieve Pretrained Backbone
    logger.info("Fetching pre-trained model from ClearML (ID: %s)...", args.clearml_id)
    try:
        clearml_model = ClearMLModel(model_id=args.clearml_id)
        checkpoint_path = clearml_model.get_local_copy()
        if not checkpoint_path:
            raise ValueError(f"ClearML returned an empty path for model ID: {args.clearml_id}")
    except Exception as e:
        logger.error("Failed to retrieve model from ClearML. Please ensure your credentials and Model ID are valid.")
        raise e
        
    logger.info("Model downloaded to: %s", checkpoint_path)
    
    # Initialize Backbone using MolGINE which accepts 38-dim continuous features
    backbone = MolGINE(
        node_input_dim=38,
        edge_input_dim=13,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        dropout=args.dropout,
        global_features_dim=0,
        output_dim=1,
    )
    
    # Load and filter state dict
    logger.info("Loading pre-trained weights into GINE backbone...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle Lightning state_dict vs PyTorch raw state_dict
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Strip any prefixes like 'model.backbone.' or 'model.' and remove 'prediction_head'
    filtered_dict = {}
    for k, v in state_dict.items():
        if "prediction_head" in k:
            continue
            
        # Strip potential lightning/wrapper prefixes
        if k.startswith("model.backbone."):
            k = k[15:]
        elif k.startswith("backbone."):
            k = k[9:]
        elif k.startswith("model."):
            k = k[6:]
            
        filtered_dict[k] = v
        
    missing_keys, unexpected_keys = backbone.load_state_dict(filtered_dict, strict=False)
    logger.info("Weights loaded. Missing keys: %d, Unexpected keys: %d", len(missing_keys), len(unexpected_keys))
    if unexpected_keys:
        logger.warning("Unexpected keys: %s", unexpected_keys[:5])
        
    # 3. Build the Hybrid Embedder
    logger.info("Wrapping backbone in CausalContrastiveUncertaintyEmbedder...")
    model = CausalContrastiveUncertaintyEmbedder(
        backbone=backbone,
        backbone_out_dim=backbone.out_channels,
        num_datasets=len(target_names),
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
        contrastive_temp=args.contrastive_temp
    )
    
    # 4. Setup Lightning Module
    lit_module = CausalSemiSupModule(
        model=model,
        task_types=task_types,
        dataset_names=target_names,
        learning_rate=args.lr,
        sparsity_beta=args.sparsity_beta,
        env_beta=args.env_beta,
        target_to_ds=target_to_ds,
        model_config={"backbone_name": "gine_finetune"}
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_target_score",
        mode="max",
        save_top_k=1,
        filename="causal_judge_best",
        save_last=True,
        every_n_epochs=10
    )
    
    finetuning_callback = BackboneWarmupCallback(
        unfreeze_epoch=args.unfreeze_epoch,
        lr_multiplier=0.1
    )
    
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-3)
    
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name="causal_finetune_gine"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=2,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, finetuning_callback, swa_callback],
        logger=tb_logger,
        enable_progress_bar=True,
    )
    
    logger.info("Starting causal fine-tuning training loop...")
    trainer.fit(
        lit_module, 
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
    )
    
    logger.info("Evaluating optimal bottleneck states...")
    ckpt_to_test = "best"
    if not checkpoint_callback.best_model_path or not Path(checkpoint_callback.best_model_path).exists():
        logger.warning("No 'best' checkpoint found. Falling back to the 'last' saved state.")
        ckpt_to_test = "last"
        
    trainer.test(lit_module, datamodule=datamodule, ckpt_path=ckpt_to_test, verbose=False)


if __name__ == "__main__":
    main()
