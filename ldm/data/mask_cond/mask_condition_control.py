from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from ldm.util import instantiate_from_config
import argparse
from omegaconf import OmegaConf
import pandas as pd
import torch

#Only mainly used if to be trained with controlnet with single control condition as finetuning on toop of T2I model.
def save_image(t, f):
    from PIL import Image
    if t.shape[-1] != 3:
        img = t.permute(1, 2, 0)
    else:
        img = t.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if np.max(img) <= 1 and np.min(img) >= -1:
        img = ((img + 1.0) * 127.5)
    Image.fromarray(img.astype(np.uint8)).save(f)
    
class MaskedDataset(Dataset):
    """Dataset with masked image as condition"""

    def __init__(self, config=None):
        split = config.get("split")
        self.split = split
        self.data_dir = Path(config.get("root"))
        self.crop_size = config.get("crop_size", None)
        self.p_uncond = config.get("p_uncond", 0)
        
        # Load three data folds and merge them
        self.data = self.read_data(self.data_dir, split)
        
    def __len__(self):
        pass

    def read_data():
        pass

    @staticmethod
    def get_random_crop(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        img = img[y : y + size, x : x + size]
        return img

    @staticmethod
    def get_central_crop(img, size):
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        half_size = size // 2
        img = img[center_y - half_size: center_y + half_size, center_x - half_size: center_x + half_size]
        return img

    def __getitem__(self, idx):   
        pass

class PanNukeDataset(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.indices = list(range(len(self.data['images'])))

    def read_data(self, data_dir, split):
        images = np.load(os.path.join(data_dir, f"{split}_images.npy"), mmap_mode='r')
        masks = np.load(os.path.join(data_dir, f"{split}_masks.npy"), mmap_mode='r')
        types = np.load(os.path.join(data_dir, f"{split}_types.npy"), mmap_mode='r')
        return {"images": images, "masks": masks, 'types': types}

    def map_values(self, x, t='identity', **kwargs):
        if t == 'identity':
            return x
        elif t == 'mult':
            return x * kwargs['m'] # 0 to 6 is mapped to 0 to 240.

    def apply_colormap_to_mask(self, mask, colormap=cv2.COLORMAP_JET, normalize=True):
        assert len(mask.shape) == 2, "Input mask must be a 2D array"
        if normalize:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask, colormap)
        assert len(colored_mask.shape) == 3, "Output mask must be a 3D array"
        return colored_mask


    def __getitem__(self, idx):
        image = self.data["images"][idx]
        mask = self.data["masks"][idx]
        type = self.data["types"][idx]
        if self.crop_size:
            mask = self.get_central_crop(mask, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32)
        mask = np.array(mask, dtype=np.uint8)
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            processed_image = np.flip(processed_image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() < 0.5:
            processed_image = np.flip(processed_image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # We need to change the mask to have it the numbers that are evenly distributed in the range of 0 to 255.
        # map_mask = np.vectorize(self.map_values)
        # mask = map_mask(mask, t='identity')   
        mask = mask.astype(np.uint8)   
        mask = self.apply_colormap_to_mask(mask, normalize=True) # normalize and convert to RGB.   
        # mask = self.apply_fixed_colormap(mask) # convert to mask with fixed colors.
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            mask = np.zeros_like(mask, np.uint8)
            
        return {
            "jpg": processed_image,
            "hint": mask,
            "txt": "",
            "type": type
        }

    def __len__(self):
        return len(self.indices)

class PanNukeDatasetV2(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.indices = list(range(len(self.data['fnames'])))
        self.inference = config.get("inference", False)
        
    def read_data(self, data_dir, split):
        data = {}
        self.split_folder = 'fold2' if self.split == 'train' else 'fold4'
        data['fnames'] = [f.split('.')[0] for f in os.listdir(os.path.join(self.data_dir, self.split_folder, 'images'))]
        data['types'] = pd.read_csv(os.path.join(self.data_dir, self.split_folder,'types.csv'), index_col='img')
        data['cell_counts'] = pd.read_csv(os.path.join(self.data_dir, self.split_folder,'cell_count.csv'), index_col='Image')
        return data

    def map_values(self, x, t='identity', **kwargs):
        if t == 'identity':
            return x
        elif t == 'mult':
            return x * kwargs['m'] # 0 to 6 is mapped to 0 to 240.

    def apply_colormap_to_mask(self, mask, colormap=cv2.COLORMAP_JET, normalize=True):
        assert len(mask.shape) == 2, "Input mask must be a 2D array"
        if normalize:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask, colormap)
        assert len(colored_mask.shape) == 3, "Output mask must be a 3D array"
        return colored_mask
    
    def apply_fixed_colormap(self, mask):
        assert len(mask.shape) == 2, "Input mask must be a 2D array"
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for value, color in self.value_to_color_map.items():
            colored_mask[mask == value] = color
        return colored_mask

    def __getitem__(self, idx):
        fname = self.data["fnames"][idx]
        image = cv2.imread(os.path.join(self.data_dir, self.split_folder, 'images', fname+".png"))
        mask = np.load(os.path.join(self.data_dir, self.split_folder, 'labels', fname+".npy"), allow_pickle=True).item()
        inst_mask = mask['inst_map']
        tissue_type = self.data['types'].loc[fname+".png", 'type']
        cell_counts = self.data['cell_counts'].loc[fname+".png", :].values

        if self.crop_size:
            inst_mask = self.get_central_crop(inst_mask, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32)
        inst_mask = np.array(inst_mask, dtype=np.uint8)
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            inst_mask = np.flip(inst_mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            inst_mask = np.flip(inst_mask, axis=1).copy()
        
        # We need to change the mask to have it the numbers that are evenly distributed in the range of 0 to 255.
        # map_mask = np.vectorize(self.map_values)
        # mask = map_mask(mask, t='identity')   
        inst_mask = inst_mask.astype(np.uint8)   
        inst_mask = self.apply_colormap_to_mask(inst_mask, normalize=True) # normalize and convert to RGB.   
        # mask = self.apply_fixed_colormap(mask) # convert to mask with fixed colors.
        
        # save_image(processed_image, f"processed_image{fname}.jpg")
        # save_image(inst_mask, f"inst_mask{fname}.jpg")  
        
        if np.random.rand() < self.p_uncond and not self.inference: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            inst_mask = np.zeros_like(inst_mask, np.uint8)
        
        ret = {
                "jpg": processed_image,
                "hint": inst_mask,
                "txt":"",
                "type": tissue_type, 
                "cell_counts": cell_counts,
                "fname": fname
            }
        return ret

    def __len__(self):
        return len(self.indices)
    
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    item = data.datasets['train'][1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)