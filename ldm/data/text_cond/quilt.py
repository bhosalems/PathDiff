from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import io
import pandas as pd
import os


class QuiltDataset(Dataset):
    """Dataset with text labels which can be used as labels of Quilt1M."""

    def __init__(self, config) -> None:
        super().__init__()
        self.split = config.get('split')
        data_dir = Path(config.get('root'))
        self.crop_size = config.get('crop_size')
        
        self.p_uncond = config.get('p_uncond', 0)
        
        # load the lookup file
        self.lookup = pd.read_csv(os.path.join(data_dir, 'quilt_1M_lookup_preprocessed.csv'))
        self.lookup = self.lookup[self.lookup['split'] == self.split]
        
        # imgdir
        self.img_path = os.path.join(data_dir, "quilt_1m")
        self.cond_col = config.get("cond_column")
        
    def __len__(self):
        # if self.split == 'train': # TODO Mahesh: comment this line, only for debugging and running on small dataset
        #     return 500
        # else:
        #     return 100
        return self.lookup.shape[0]
    
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
        sample_df = self.lookup.iloc[idx]
        img_path = os.path.join(self.img_path, sample_df['image_path'])
        tile = Image.open(img_path)
        tile = tile.convert('RGB') # convert all images to by deault RGB.
        tile = np.array(tile)
        if tile.shape[2] !=3 or len(tile.shape) != 3:
            print(f"Image {img_path} has shape {tile.shape}")
            return None
             
        
        
        image = (tile / 127.5 - 1.0).astype(np.float32)
        if self.crop_size:
            # image = self.get_random_crop(image, self.crop_size)
            image = self.get_central_crop(image, self.crop_size) # For quilt dataset we should get central crop instead of random crop.
        
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
        
        text_prompt = sample_df[self.cond_col]
        if text_prompt is np.nan: # if condition text is nan, we have to use another condition text
            text_prompt = sample_df['corrected_text']
            
        # Replace text prompt with unconditional text prompt with probability p_uncond
        if np.random.rand() < self.p_uncond:
            text_prompt = ""   
        return {
            "image": image,
            "caption": text_prompt
        }
        
        