from torch.utils.data import Dataset
from ldm.data.text_cond.pathcap import PathcapDataset
from ldm.data.text_cond.tumor_til_in_text import TCGADataset
from ldm.data.mask_cond.mask_condition import PanNukeDataset, PanNukeDatasetV2, SegPathDataset, ConicDataset, MonuSacDataset
import numpy as np
from ldm.data.mask_cond.mask_condition import NULL_MASK

class ControlnetWrapperDataset(Dataset):
    """Dataset for mixed conditions"""
    def __init__(self, config):
        self.datasets = {}
        self.configs = config['datasets']
        self.dataset_sizes = {}
        self.hint_channels = config.get("hint_channels", 3)
        dataset_classes = {
            "PanNuke": PanNukeDataset, # "M2I" Mask to image
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
                    dataset_config = self.configs[key]['config']
                    dataset_config['inference'] = self.inference
                    dataset_config['hint_channels'] = self.hint_channels
                    self.datasets[key] = dataset_class(dataset_config)   
                    self.dataset_sizes[key] = len(self.datasets[key])
        self.split_prob = 0
        self.data_size = min(self.dataset_sizes.values()) # doesnt matter, just one value
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        caption = "Histopathology image"
        if not self.inference:
            data = self.datasets[self.m2i][index % self.dataset_sizes[self.m2i]]
            return {
            "jpg": data['image'],
            "hint": data['mask'],
            "txt": caption
            }
        else:
            data = self.datasets[self.m2i][index % self.dataset_sizes[self.m2i]]
            data["mask"] = data["mask"]
            ret = {
                "jpg": data['image'],
                "jpg2": data['image'],
                "hint": data['mask'],
                "txt": caption,
                "fname": "",
                "fname2": data['fname'],
            }
            if 'type' in data:
                ret['type'] = data['type']
            if 'cell_counts' in data:
                ret['cell_counts']  = data['cell_counts']
            
            # print([type(ret[k]) for k in ret.keys()])
            return ret

class ControlnetWrapperDatasetv2(Dataset): # used for inferencrence of Controlnet-T
    """Dataset for mixed conditions"""
    def __init__(self, config):
        self.datasets = {}
        self.configs = config['datasets']
        self.dataset_sizes = {}
        self.hint_channels = config.get("hint_channels", 3)
        dataset_classes = {
            "PathCap": PathcapDataset, # "T2I" Text to Image
        }
        self.inference = config.get("inference", False)
        for key, dataset_class in dataset_classes.items():
            if key in self.configs:
                if key in ["PathCap"]:
                    self.t2i = key
                    dataset_config = self.configs[key]['config']
                    dataset_config['inference'] = self.inference
                    dataset_config['hint_channels'] = self.hint_channels
                    self.datasets[key] = dataset_class(dataset_config)   
                    self.dataset_sizes[key] = len(self.datasets[key])
        self.data_size = min(self.dataset_sizes.values()) # doesnt matter, just one value
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        if self.inference:
            data = self.datasets[self.t2i][index % self.dataset_sizes[self.t2i]]
            data["mask"] = np.full((256, 256, self.hint_channels), NULL_MASK, dtype=np.uint8)
            ret = {
                "jpg": data['image'],
                "jpg2": data['image'],
                "hint": data['mask'],
                "txt": data['caption'],
                "fname": data['fname'],
                "fname2": "",
            }
            return ret
