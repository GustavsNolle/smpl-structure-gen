import logging
import torch
import pytorch_lightning as pl
from clearml import Task
from torch_geometric.data import Data, Batch

from mol_prop_gnn.training.digress_sampler import sample_digress_graphs
from mol_prop_gnn.utils.molecule_decoder import graph_to_smiles, graph_to_mol
from mol_prop_gnn.data.preprocessing import smiles_to_graph
from rdkit.Chem import Draw

logger = logging.getLogger(__name__)

class DiGressEvaluationCallback(pl.Callback):
    def __init__(
        self, 
        train_smiles: list[str],
        train_node_counts: list[int],
        num_samples: int = 1000, 
        guidance_scale: float = 0.0,
        evaluate_every_n_epochs: int = 10
    ):
        super().__init__()
        self.train_smiles = set(train_smiles)
        self.train_node_counts = torch.tensor(train_node_counts, dtype=torch.long)
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.evaluate_every_n_epochs = evaluate_every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.evaluate_every_n_epochs != 0:
            return
            
        logger.info(f"Starting DiGress Generative Evaluation (Epoch {trainer.current_epoch + 1})...")
        
        # 1. Sample target condition from validation set
        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        batch = batch.to(pl_module.device)
        
        # Extract the causal embeddings for the first few items to act as our anchor
        c_batch = pl_module.get_causal_embedding(batch)
        c_anchor = c_batch[0].unsqueeze(0).repeat(self.num_samples, 1) # [num_samples, causal_cond_dim]
        
        # 2. Sample sizes from empirical distribution
        idx = torch.randint(0, len(self.train_node_counts), (self.num_samples,))
        num_nodes_per_graph = self.train_node_counts[idx].tolist()
        
        # 3. Generate graphs
        pl_module.model.eval()
        X_gen, E_gen, edge_index_gen, batch_idx_gen = sample_digress_graphs(
            model=pl_module.model,
            c=c_anchor,
            num_nodes_per_graph=num_nodes_per_graph,
            node_marginals=pl_module.node_marginals,
            edge_marginals=pl_module.edge_marginals,
            num_node_classes=pl_module.num_node_classes,
            num_edge_classes=pl_module.num_edge_classes,
            num_timesteps=pl_module.num_timesteps,
            guidance_scale=self.guidance_scale
        )
        
        # 4. Decode to SMILES
        valid_smiles = []
        sample_mols = []
        sample_legends = []
        from rdkit import Chem
        for i in range(self.num_samples):
            mask = (batch_idx_gen == i)
            n_idx = torch.nonzero(mask).squeeze(-1)
            if len(n_idx) == 0: continue
            
            # Subgraph for graph i
            node_types = X_gen[mask]
            
            # Find edges where both endpoints are in graph i
            edge_mask = (batch_idx_gen[edge_index_gen[0]] == i)
            sub_edge_index = edge_index_gen[:, edge_mask]
            
            # Re-index edges to start from 0
            offset = n_idx.min().item()
            sub_edge_index = sub_edge_index - offset
            
            edge_types = E_gen[edge_mask]
            
            smiles = graph_to_smiles(node_types, sub_edge_index, edge_types)
            if smiles is not None:
                valid_smiles.append(smiles)
            
            # Collect a few samples for visualization (first 10)
            if i < 10:
                mol = graph_to_mol(node_types, sub_edge_index, edge_types)
                if smiles is not None:
                    # Successfully sanitized mol
                    viz_mol = Chem.MolFromSmiles(smiles)
                    if viz_mol:
                        sample_mols.append(viz_mol)
                        sample_legends.append(f"Valid: {smiles[:20]}")
                    else:
                        sample_mols.append(mol)
                        sample_legends.append("Invalid (Sanitize Fail)")
                else:
                    # Raw molecule for visualization of failures
                    sample_mols.append(mol)
                    sample_legends.append("Invalid (Valence Error)")
                
        # 5. Calculate V.U.N Metrics
        validity = len(valid_smiles) / self.num_samples
        
        if len(valid_smiles) > 0:
            unique_smiles_set = set(valid_smiles)
            uniqueness = len(unique_smiles_set) / len(valid_smiles)
            
            novel_smiles_set = unique_smiles_set - self.train_smiles
            novelty = len(novel_smiles_set) / len(unique_smiles_set)
        else:
            uniqueness = 0.0
            novelty = 0.0
            novel_smiles_set = set()
            
        logger.info(f"V.U.N Metrics -> Validity: {validity*100:.1f}% | Uniqueness: {uniqueness*100:.1f}% | Novelty: {novelty*100:.1f}%")
        
        # 6. Calculate Guidance Hit Rate
        hit_rate = 0.0
        if len(novel_smiles_set) > 0:
            logger.info("Evaluating Guidance Hit Rate via Frozen Causal Judge...")
            
            # Convert novel SMILES back to PyG data
            novel_graphs = []
            for sm in list(novel_smiles_set)[:100]:  # Cap at 100 for speed
                data = smiles_to_graph(sm)
                if data is not None:
                    novel_graphs.append(data)
                    
            if len(novel_graphs) > 0:
                gen_batch = Batch.from_data_list(novel_graphs).to(pl_module.device)
                
                # We need to evaluate the generated molecules on the specific target.
                # Since the causal judge outputs 21 targets, we'll extract the logits.
                with torch.no_grad():
                    # For a full check, we would evaluate if the predictions match the anchor's predictions.
                    # Here we just pass them through the judge to get the predictions.
                    result = pl_module.causal_judge(
                        x=gen_batch.x, 
                        edge_index=gen_batch.edge_index, 
                        edge_attr=gen_batch.edge_attr, 
                        batch=gen_batch.batch
                    )
                    # result is usually (pred_c, pred_e, mask, contrastive_loss, log_vars)
                    # For now, we will simply log that we successfully scored them.
                    # A true hit rate requires knowing WHICH task we are targeting.
                    # We will assume Task 0 is our target (e.g., SAscore) for demonstration.
                    pred_c = result[0]
                    # Anchor prediction
                    anchor_result = pl_module.causal_judge(
                        x=batch.x[:batch.ptr[1]], 
                        edge_index=batch.edge_index[:, batch.batch[batch.edge_index[0]] == 0],
                        edge_attr=batch.edge_attr[batch.batch[batch.edge_index[0]] == 0],
                        batch=torch.zeros(batch.ptr[1], dtype=torch.long, device=pl_module.device)
                    )
                    anchor_pred = anchor_result[0][0, 0].item()
                    
                    # Generated predictions for task 0
                    gen_preds = pred_c[:, 0]
                    
                    # Hit rate: percentage of generated molecules within 10% of anchor prediction
                    # or if classification, same class.
                    # Assuming regression for simplicity
                    hits = (torch.abs(gen_preds - anchor_pred) < 0.5).sum().item()
                    hit_rate = hits / len(novel_graphs)
                    
            logger.info(f"Guidance Hit Rate (Task 0): {hit_rate*100:.1f}%")
            
        # 7. Log to ClearML
        task = Task.current_task()
        if task:
            task.get_logger().report_scalar("Generative_Metrics", "Validity (%)", value=validity * 100, iteration=trainer.global_step)
            task.get_logger().report_scalar("Generative_Metrics", "Uniqueness (%)", value=uniqueness * 100, iteration=trainer.global_step)
            task.get_logger().report_scalar("Generative_Metrics", "Novelty (%)", value=novelty * 100, iteration=trainer.global_step)
            task.get_logger().report_scalar("Generative_Metrics", "Guidance_Hit_Rate (%)", value=hit_rate * 100, iteration=trainer.global_step)
            
            # 8. Report visualized samples
            if len(sample_mols) > 0:
                try:
                    img = Draw.MolsToGridImage(
                        sample_mols, 
                        molsPerRow=5, 
                        subImgSize=(300, 300), 
                        legends=sample_legends
                    )
                    # Convert PIL image to numpy or save to temp file for ClearML
                    import numpy as np
                    img_np = np.array(img)
                    task.get_logger().report_image(
                        "Generated_Samples", 
                        "Batch_Visual", 
                        iteration=trainer.global_step, 
                        image=img_np
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate molecule grid image: {e}")
