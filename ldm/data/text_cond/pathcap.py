from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os


class PathcapDataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        split = config.get("split")
        self.split = split
        self.data_dir = Path(config.get("root"))
        self.crop_size = config.get("crop_size", None)
        self.inference = config.get("inference", False)

        self.p_uncond = config.get("p_uncond", 0)

        # Load the annotations file
        self.annot = pd.read_csv(os.path.join(self.data_dir, f"pathcap_{split}.csv")) 

    def __len__(self):
        return len(self.annot)

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

    @staticmethod
    def resize_with_aspect_ratio(tile, target_size):
        w, h = tile.size
        if h < target_size or w < target_size:
            aspect_ratio = w / h
            if h < w:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            else:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            tile = tile.resize((new_w, new_h), Image.BICUBIC)
        return tile


    def __getitem__(self, idx):
        annot = self.annot.iloc[idx]

        assert annot['split'] == self.split
        tile = Image.open(os.path.join(self.data_dir, annot['img']))
        if tile.mode != 'RGB':
            tile = tile.convert("RGB")
        tile = self.resize_with_aspect_ratio(tile, self.crop_size+1)
        
        tile = np.array(tile)
        image = (tile / 127.5 - 1.0).astype(np.float32)
        
        if self.crop_size:
            # image = self.get_random_crop(image, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)

        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            image = np.flip(image, axis=1).copy()
            
        text_prompt = annot['text']

        # Replace text prompt with unconditional text prompt with probability p_uncond
        # Dont replace if p_til is positive
        if np.random.rand() < self.p_uncond:
            text_prompt = ""

        return {
            "image": image,
            "caption": text_prompt,
            "fname": annot['img'].split("/")[-1].split(".")[0]
        }