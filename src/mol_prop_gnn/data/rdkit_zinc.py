"""RDKit-featurized ZINC dataset for pretraining."""

import os
import urllib.request
from typing import Callable, List, Optional

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from mol_prop_gnn.data.preprocessing import smiles_to_graph


class RDKitZINC(InMemoryDataset):
    """ZINC dataset processed with continuous/one-hot RDKit features.
    
    This matches the exact feature dimensions (38 for nodes, 13 for edges)
    used in downstream multi-task supervised learning, preventing architecture
    mismatches during Phase 2 finetuning.
    """
    
    # URL for ZINC 250k subset
    url = "https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/train.txt"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return ["zinc_250k.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["rdkit_zinc_250k.pt"]

    def download(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading {self.url}...")
            urllib.request.urlretrieve(self.url, raw_path)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, 'r') as f:
            smiles_list = f.read().splitlines()
            
        # The file might have a header or just smiles
        if "smiles" in smiles_list[0].lower():
            smiles_list = smiles_list[1:]
            
        data_list = []
        for s in tqdm(smiles_list, desc="Processing RDKit ZINC"):
            s = s.strip()
            if not s:
                continue
                
            data = smiles_to_graph(s)
            if data is not None and data.x.shape[0] > 0:
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
