from torch.utils.data import Dataset
from ldm.data.text_cond.pathcap import PathcapDataset
from ldm.data.text_cond.tumor_til_in_text import TCGADataset
from ldm.data.mask_cond.mask_condition import PanNukeDataset, PanNukeDatasetV2, SegPathDataset, ConicDataset, MonuSacDataset
import numpy as np
from ldm.data.mask_cond.mask_condition import NULL_MASK

class MixedConditionDataset(Dataset):
    """Dataset for mixed conditions"""
    def __init__(self, config):
        self.datasets = {}
        self.configs = config['datasets']
        self.dataset_sizes = {}
        self.hint_channels = config.get("hint_channels", 3)
        dataset_classes = {
            "PathCap": PathcapDataset, # "T2I" Text to Image
            "PanNuke": PanNukeDataset, # "M2I" Mask to image
            "TCGA": TCGADataset, # "T2T"
            "PanNukeV2": PanNukeDatasetV2, # "M2I"          
            "SegPath": SegPathDataset, # "M2I" Mask to image
            "Conic": ConicDataset, # "M2I" Mask to image
            "Monusac": MonuSacDataset # "M2I" Mask to image
        }
        self.inference = config.get("inference", False)
        for key, dataset_class in dataset_classes.items():
            if key in self.configs:
                if key in ["PanNuke", "PanNukeV2", "SegPath", "Conic", "Monusac"]:
                    self.m2i = key
                else:
                    self.t2i = key
                dataset_config = self.configs[key]['config']
                dataset_config['inference'] = self.inference
                dataset_config['hint_channels'] = self.hint_channels
                self.datasets[key] = dataset_class(dataset_config)   
                self.dataset_sizes[key] = len(self.datasets[key])
        self.split_prob = config.get("split_prob", 0.5)
        if self.inference:
            self.data_size = min(self.dataset_sizes.values()) # In current setting we only use the smallest data condition once.
        else:
            if self.split_prob == 0:
                self.data_size = self.dataset_sizes[self.m2i]
            elif self.split_prob == 1:
                self.data_size = self.dataset_sizes[self.t2i]
            else:
                self.data_size = max(self.dataset_sizes.values())
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        if not self.inference:
            if np.random.rand() <= self.split_prob: # if split_prob=0 then only mask to image, if split_prob=1 then only text to image
                data = self.datasets[self.t2i][index % self.dataset_sizes[self.t2i]]
                data["mask"] = np.full((256, 256, self.hint_channels), NULL_MASK, dtype=np.uint8) 
            else:
                data = self.datasets[self.m2i][index % self.dataset_sizes[self.m2i]]
                data['caption'] = ""
            return {
            "jpg": data['image'],
            "hint": data['mask'],
            "txt": data['caption']
            }
        else:
            data = self.datasets[self.t2i][index % self.dataset_sizes[self.t2i]]
            data2 = self.datasets[self.m2i][index % self.dataset_sizes[self.m2i]]
            data["mask"] = data2["mask"]
            ret = {
                "jpg": data['image'],
                "jpg2": data2['image'],
                "hint": np.full((256, 256, self.hint_channels), NULL_MASK, dtype=np.uint8), # data['mask'], 
                "txt": data['caption'], # ""
                "fname": data['fname'],
                "fname2": data2['fname']    
            }
            if 'type' in data2:
                ret['type'] = data2['type']
            if 'cell_counts' in data2:
                ret['cell_counts']  = data2['cell_counts']
            return ret
