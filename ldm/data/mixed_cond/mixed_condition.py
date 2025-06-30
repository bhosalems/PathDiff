from torch.utils.data import Dataset
from ldm.data.text_cond.tumor_til_in_text import TCGADataset
from ldm.data.mask_cond.mask_condition import PanNukeDatasetV2
import numpy as np

class MixedConditionDataset(Dataset):
    """Dataset for mixed conditions"""
    def __init__(self, config):
        self.datasets = {}
        self.configs = config['datasets']
        self.dataset_sizes = {}
        dataset_classes = {
            "TCGA": TCGADataset,
            "PanNuke": PanNukeDatasetV2
        }
        for key, dataset_class in dataset_classes.items():
            if key in self.configs:
                self.datasets[key] = dataset_class(self.configs[key]['config'])   
                self.dataset_sizes[key] = len(self.datasets[key])
        self.split_prob = config.get("split_prob", 0.5) 
    
    def __len__(self):
        # return min(self.dataset_sizes.values()) # May be we need to adjust the sizes of the datasets here since there are a lot of such datasets
        return sum(self.dataset_sizes.values())
    
    def __getitem__(self, index):
        if np.random.rand() <= self.split_prob:
            data = self.datasets["TCGA"][index % self.dataset_sizes["TCGA"]]
            data["mask"] = np.zeros((256, 256, 3), np.uint8)
        else:
            data = self.datasets["PathCap"][index % self.dataset_sizes["PathCap"]]
        return data
