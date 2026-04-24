import logging
import sys
import os
import torch
import pytorch_lightning as pl
from clearml import Task
from torch_geometric.data import Data, Batch

from mol_prop_gnn.training.digress_sampler import sample_digress_graphs
from mol_prop_gnn.utils.molecule_decoder import graph_to_smiles, graph_to_mol
from mol_prop_gnn.data.preprocessing import smiles_to_graph
from rdkit import Chem
from rdkit.Chem import Draw, RDConfig

# Import RDKit's SA Score calculator
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

logger = logging.getLogger(__name__)

# SA Score is task index 8 in the Causal Judge's output.
# The Judge was trained on: ["bace", "bbbp", "clintox"(2), "esol", "freesolv",
#                            "herg", "lipophilicity", "sascore", "tox21"(12)]
# After alphabetical sorting and multi-task expansion:
#   0:bace, 1:bbbp, 2-3:clintox, 4:esol, 5:freesolv,
#   6:herg, 7:lipophilicity, 8:sascore, 9-20:tox21
SASCORE_TASK_IDX = 8


def compute_actual_sa(smiles: str) -> float | None:
    """Compute actual SA score (1-10 scale) from SMILES using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return sascorer.calculateScore(mol)
    except Exception:
        pass
    return None


def estimate_sa_scaling(smiles_list: list[str]) -> tuple[float, float]:
    """Estimate mean/std of SA scores from a list of SMILES.
    
    Used to invert the Judge's z-scored SA predictions back to the 1-10 scale.
    """
    scores = []
    for smi in smiles_list[:200]:  # Cap for speed
        sa = compute_actual_sa(smi)
        if sa is not None:
            scores.append(sa)
    
    if len(scores) < 5:
        # Fallback: typical ZINC SA score distribution
        return 3.05, 0.83
    
    import numpy as np
    return float(np.mean(scores)), float(np.std(scores))


def unscale_sa(z_score: float, mean: float, std: float) -> float:
    """Convert z-scored SA prediction back to 1-10 scale."""
    raw = z_score * std + mean
    return max(1.0, min(10.0, raw))  # Clamp to valid SA range


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
        self.train_smiles_list = list(train_smiles)[:200]  # Keep a small list for SA scaling
        self.train_node_counts = torch.tensor(train_node_counts, dtype=torch.long)
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        
        # Compute SA scaling stats once from training SMILES
        self._sa_mean = None
        self._sa_std = None

    def _get_sa_scaling(self) -> tuple[float, float]:
        """Lazy-load SA scaling stats from training data."""
        if self._sa_mean is None:
            logger.info("Computing SA score scaling stats from training SMILES...")
            self._sa_mean, self._sa_std = estimate_sa_scaling(self.train_smiles_list)
            logger.info(f"SA scaling: mean={self._sa_mean:.3f}, std={self._sa_std:.3f}")
        return self._sa_mean, self._sa_std

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.evaluate_every_n_epochs != 0:
            return
        
        try:
            self._run_evaluation(trainer, pl_module)
        except Exception as e:
            logger.error(f"DiGress Evaluation FAILED (non-fatal): {e}", exc_info=True)
            task = Task.current_task()
            if task:
                task.get_logger().report_scalar("Generative_Metrics", "Validity (%)", value=0.0, iteration=trainer.global_step)
    
    def _get_anchor_molecule(self, trainer, pl_module):
        """Finds a molecule in the validation set with the lowest SA score.
        Low SA score = easy to synthesize. This is the guidance target.
        Returns the causal embedding, SMILES, and property dict (SA, LogP, QED).
        """
        val_dataloader = trainer.datamodule.val_dataloader()
        
        best_z_sascore = float('inf')
        best_c = None
        best_smiles = "N/A"
        best_batch = None
        best_props = {}
        
        # Scan a few validation batches to find a good anchor
        for batch_idx_iter, batch in enumerate(val_dataloader):
            if batch_idx_iter >= 3:  # Check first 3 batches (~384 molecules)
                break
                
            batch = batch.to(pl_module.device)
            
            with torch.no_grad():
                # Get Judge features
                from mol_prop_gnn.training.digress_module import digress_to_judge_features
                x_j, _ = digress_to_judge_features(batch.x.argmax(dim=-1), 
                                                  batch.edge_attr.argmax(dim=-1), 
                                                  batch.edge_index)
                
                # 1. Compute scores for the WHOLE batch first (safer and faster)
                if "sascore" in pl_module.property_judges:
                    sa_scores = pl_module.property_judges["sascore"](
                        x=x_j, edge_index=batch.edge_index, batch=batch.batch
                    )
                else:
                    result = pl_module.causal_judge(
                        x=batch.x, edge_index=batch.edge_index, 
                        edge_attr=batch.edge_attr, batch=batch.batch
                    )
                    sa_scores = result[0][:, SASCORE_TASK_IDX]
                
                logp_scores = None
                if "logp" in pl_module.property_judges:
                    logp_scores = pl_module.property_judges["logp"](
                        x=x_j, edge_index=batch.edge_index, batch=batch.batch
                    )
                    
                qed_scores = None
                if "qed" in pl_module.property_judges:
                    qed_scores = pl_module.property_judges["qed"](
                        x=x_j, edge_index=batch.edge_index, batch=batch.batch
                    )
                
                # 2. Pick the best molecule based on SA
                min_idx = sa_scores.argmin().item()
                min_sa_val = sa_scores[min_idx].item()
                
                if min_sa_val < best_z_sascore:
                    best_z_sascore = min_sa_val
                    best_c = pl_module.get_causal_embedding(batch)[min_idx].unsqueeze(0)
                    best_batch = batch
                    
                    # Store properties for the best found
                    best_props = {"sa": min_sa_val}
                    if logp_scores is not None:
                        best_props["logp"] = logp_scores[min_idx].item()
                    if qed_scores is not None:
                        best_props["qed"] = qed_scores[min_idx].item()
                    
                    if hasattr(batch, 'smiles'):
                        best_smiles = batch.smiles[min_idx] if isinstance(batch.smiles, list) else "N/A"

        
        # Log properties
        sa_str = f"SA={best_props.get('sa', 0):.2f}"
        logp_str = f"LogP={best_props.get('logp', 0):.2f}" if "logp" in best_props else ""
        qed_str = f"QED={best_props.get('qed', 0):.2f}" if "qed" in best_props else ""
        
        logger.info(f"Anchor molecule: {sa_str} {logp_str} {qed_str}, SMILES={best_smiles[:40]}")
        return best_c, best_smiles, best_props, best_batch


    
    def _run_evaluation(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info(f"Starting DiGress Generative Evaluation (Epoch {trainer.current_epoch + 1})...")
        
        # 1. Find the best low-SA-score anchor molecule
        c_anchor, anchor_smiles, anchor_props, anchor_batch = \
            self._get_anchor_molecule(trainer, pl_module)
        
        if c_anchor is None:
            logger.warning("Could not find a valid anchor molecule. Skipping evaluation.")
            return
        
        # 2. Sample sizes from empirical distribution
        idx = torch.randint(0, len(self.train_node_counts), (self.num_samples,))
        all_num_nodes = self.train_node_counts[idx].tolist()
        
        # 3. Generate graphs IN BATCHES to prevent GPU OOM
        CHUNK_SIZE = 50
        all_X, all_E, all_edge_index, all_batch_idx = [], [], [], []
        
        pl_module.model.eval()
        node_offset = 0
        batch_offset = 0
        
        for chunk_start in range(0, self.num_samples, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, self.num_samples)
            chunk_size = chunk_end - chunk_start
            
            c_chunk = c_anchor.repeat(chunk_size, 1)
            chunk_nodes = all_num_nodes[chunk_start:chunk_end]
            
            X_c, E_c, ei_c, bi_c = sample_digress_graphs(
                model=pl_module.model,
                c=c_chunk,
                num_nodes_per_graph=chunk_nodes,
                node_marginals=pl_module.node_marginals,
                edge_marginals=pl_module.edge_marginals,
                num_node_classes=pl_module.num_node_classes,
                num_edge_classes=pl_module.num_edge_classes,
                num_timesteps=pl_module.num_timesteps,
                guidance_scale=self.guidance_scale
            )
            
            all_X.append(X_c.cpu())
            all_E.append(E_c.cpu())
            all_edge_index.append(ei_c.cpu() + node_offset)
            all_batch_idx.append(bi_c.cpu() + batch_offset)
            
            node_offset += X_c.shape[0]
            batch_offset += chunk_size
            
            torch.cuda.empty_cache()
            
        X_gen = torch.cat(all_X)
        E_gen = torch.cat(all_E)
        edge_index_gen = torch.cat(all_edge_index, dim=1)
        batch_idx_gen = torch.cat(all_batch_idx)
            
        # 4. Decode into molecules
        valid_mols = [] # List of (mol, smiles, actual_sa)
        invalid_mols = [] # List of (mol, reason)
        
        for i in range(self.num_samples):
            mask = (batch_idx_gen == i)
            n_idx = torch.nonzero(mask).squeeze(-1)
            if len(n_idx) == 0:
                continue
                
            node_types = X_gen[mask].reshape(-1)
            edge_mask = (batch_idx_gen[edge_index_gen[0]] == i)
            sub_edge_index = edge_index_gen[:, edge_mask]
            offset = n_idx.min().item()
            sub_edge_index = sub_edge_index - offset
            edge_types = E_gen[edge_mask].reshape(-1)
            
            # Convert back to SMILES (this sanitizes and validates!)
            from mol_prop_gnn.utils.molecule_decoder import graph_to_smiles
            smiles = graph_to_smiles(node_types, sub_edge_index, edge_types)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    actual_sa = compute_actual_sa(smiles)
                    valid_mols.append((mol, smiles, actual_sa, node_types, sub_edge_index, edge_types))
                else:
                    invalid_mols.append((None, "RDKit Parse Error"))
            else:
                invalid_mols.append((None, "Valence Error"))
                    
        # 5. Calculate V.U.N Metrics
        validity = len(valid_mols) / max(self.num_samples, 1)
        
        if len(valid_mols) > 0:
            valid_smiles_list = [m[1] for m in valid_mols]
            unique_smiles_set = set(valid_smiles_list)
            uniqueness = len(unique_smiles_set) / len(valid_mols)
            novel_smiles_set = unique_smiles_set - self.train_smiles
            novelty = len(novel_smiles_set) / max(len(unique_smiles_set), 1)
        else:
            uniqueness = 0.0
            novelty = 0.0
            novel_smiles_set = set()
            
        logger.info(f"V.U.N Metrics -> Validity: {validity*100:.1f}% | Uniqueness: {uniqueness*100:.1f}% | Novelty: {novelty*100:.1f}%")
        
        # 6. Calculate Property stats using Mini Judges for valid molecules in the grid
        gen_props = {"sa": [], "logp": [], "qed": []}
        
        if len(valid_mols) > 0:
            logger.info("Evaluating properties via Mini Judges for visualization...")
            # We evaluate the first 16 valid molecules (the ones shown in the grid)
            grid_samples = valid_mols[:16]
            
            from mol_prop_gnn.data.preprocessing import smiles_to_graph
            
            for mol, sm, _, _, _, _ in grid_samples:
                with torch.no_grad():
                    data = smiles_to_graph(sm)
                    if data is not None:
                        xj = data.x.to(pl_module.device)
                        ei_d = data.edge_index.to(pl_module.device)
                        b_d = torch.zeros(xj.shape[0], dtype=torch.long, device=pl_module.device)
                        
                        if "sascore" in pl_module.property_judges:
                            gen_props["sa"].append(pl_module.property_judges["sascore"](x=xj, edge_index=ei_d, batch=b_d).item())
                        else:
                            gen_props["sa"].append(0.0)
                            
                        if "logp" in pl_module.property_judges:
                            gen_props["logp"].append(pl_module.property_judges["logp"](x=xj, edge_index=ei_d, batch=b_d).item())
                        else:
                            gen_props["logp"].append(0.0)
                            
                        if "qed" in pl_module.property_judges:
                            gen_props["qed"].append(pl_module.property_judges["qed"](x=xj, edge_index=ei_d, batch=b_d).item())
                        else:
                            gen_props["qed"].append(0.0)
                    else:
                        gen_props["sa"].append(0.0)
                        gen_props["logp"].append(0.0)
                        gen_props["qed"].append(0.0)

        # 7. Log to ClearML
        task = Task.current_task()
        if task:
            cl = task.get_logger()
            step = trainer.global_step
            cl.report_scalar("Generative_Metrics", "Validity (%)", value=validity * 100, iteration=step)
            cl.report_scalar("Generative_Metrics", "Uniqueness (%)", value=uniqueness * 100, iteration=step)
            cl.report_scalar("Generative_Metrics", "Novelty (%)", value=novelty * 100, iteration=step)
            
            # 8. Visualize: Anchor + 16 Valid + 8 Invalid
            self._report_sample_grid(
                task, trainer, valid_mols, invalid_mols,
                anchor_smiles, anchor_props, gen_props
            )

    
    def _report_sample_grid(self, task, trainer, valid_mols, invalid_mols,
                            anchor_smiles, anchor_props, gen_props):
        """Reports a grid: Anchor row + 16 Valid + 8 Invalid molecules.
        Displays SA, LogP, and QED metrics in the legends.
        """
        import numpy as np
        from rdkit.Chem import Draw
        
        cl = task.get_logger()
        step = trainer.global_step
        
        grid_mols = []
        grid_legends = []
        
        # 1. The Anchor molecule
        anchor_mol = Chem.MolFromSmiles(anchor_smiles) if anchor_smiles != "N/A" else None
        if anchor_mol:
            s = anchor_props.get("sa", 0)
            l = anchor_props.get("logp", 0)
            q = anchor_props.get("qed", 0)
            grid_mols.append(anchor_mol)
            grid_legends.append(f"ANCHOR S:{s:.1f} L:{l:.1f} Q:{q:.1f}")
        
        # 2. Up to 16 VALID generated molecules
        for idx, item in enumerate(valid_mols[:16]):
            mol, smi, actual_sa = item[0], item[1], item[2]

            s = gen_props["sa"][idx] if idx < len(gen_props["sa"]) else 0
            l = gen_props["logp"][idx] if idx < len(gen_props["logp"]) else 0
            q = gen_props["qed"][idx] if idx < len(gen_props["qed"]) else 0
            actual_sa_str = f"({actual_sa:.1f})" if actual_sa is not None else ""
            
            grid_mols.append(mol)
            grid_legends.append(f"V S:{s:.1f}{actual_sa_str} L:{l:.1f} Q:{q:.2f}")
        
        # Pad to keep grid aligned if fewer than 16 valid
        while len(grid_mols) < 17: # 1 anchor + 16 valid
            grid_mols.append(Chem.MolFromSmiles("C"))
            grid_legends.append("(no valid sample)")
        
        # 3. Up to 8 INVALID generated molecules  
        for mol, reason in invalid_mols[:8]:
            if mol is not None:
                grid_mols.append(mol)
            else:
                grid_mols.append(Chem.MolFromSmiles("C"))
            grid_legends.append(f"INV: {reason}")
            
        # Draw grid (4 columns)
        img = Draw.MolsToGridImage(
            grid_mols, 
            molsPerRow=4, 
            subImgSize=(300, 300), 
            legends=grid_legends
        )

        
        cl.report_image(
            title="Generated_Samples_Grid",
            series="Multi-Property Guidance",
            iteration=step,
            image=img
        )

