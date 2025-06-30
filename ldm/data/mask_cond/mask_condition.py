from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import torch 
import random
from PIL import Image

def save_image(t, f, normalize=False):
    if t.shape[-1] != 3 and isinstance(t, torch.Tensor):
        img = t.permute(1, 2, 0)
    else:
        img = t.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # # Handle normalization to [0, 255]
    if normalize and np.max(img) <= 1 and np.min(img) >= -1:
        if np.max(img) == np.min(img):  # This means the array is all zeros or a constant
            img = np.zeros_like(img)  # Set all values to 0 (black)
        else:
            img = ((img + 1.0) * 127.5)

    Image.fromarray(img.astype(np.uint8)).save(f)

FIXED_COLOR_MAP = {
    0: (0, 0, 0),          # Cell type 0 - Black
    1: (255, 0, 0),        # Cell type 1 - Red
    2: (0, 255, 0),        # Cell type 2 - Green
    3: (0, 0, 255),        # Cell type 3 - Blue
    4: (255, 255, 0),      # Cell type 4 - Yellow
    5: (255, 0, 255),      # Cell type 5 - Magenta
    6: (0, 255, 255),      # Cell type 6 - Cyan
    7: (255, 165, 0),      # Cell type 7 - Orange
    8: (128, 128, 128),    # Cell type 8 - Gray
    9: (255, 255, 255),     # Cell type 9 - White (For edges)
}

# For label prediction
# Class 1: Lymphocytes – Red → RGB: [255, 0, 0]
# Class 2: Epithelial cells – Green → RGB: [0, 255, 0]
# Class 3: Plasma cells – Blue → RGB: [0, 0, 255]
# Class 4: Neutrophils – Yellow → [255, 255, 0]
# Class 5: Eosinophils – Magenta → RGB: [255, 0, 255]
# Class 6: Connective tissue – Cyan → RGB: [255, 255, 0]

# CONIC: { 'neutrophil': 1, 'epithelial': 2, 'lymphocyte': 3,
#                                       'plasma': 4, 'eosinophil': 5, 'connective' : 6}


NULL_MASK = 10 

# FIXED_COLOR_MAP = {
#     0: (0, 0, 0),          # Cell type 0 - Black
#     1: (50, 50, 50),        # Cell type 1 - Red
#     2: (75, 75, 75),        # Cell type 2 - Green
#     3: (100, 100, 100),        # Cell type 3 - Blue
#     4: (125, 125, 125),      # Cell type 4 - Yellow
#     5: (150, 150, 150),      # Cell type 5 - Magenta
#     6: (175, 175, 175),      # Cell type 6 - Cyan
#     7: (200, 200, 200),      # Cell type 7 - Orange
#     8: (210, 210, 210),    # Cell type 8 - Gray
#     9: (255, 255, 255)     # Cell type 9 - White (For edges)
# }

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
    def apply_colormap_to_mask(mask, class_names_to_color=None, normalize=True, colormap=cv2.COLORMAP_JET):
        assert isinstance(mask, np.ndarray), "Input mask must be a numpy array"
        assert len(mask.shape) == 2, "Input mask must be a 2D array"
        if class_names_to_color:
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            # Apply the fixed colormap based on the label values
            for label, color in class_names_to_color.items():
                colored_mask[mask == label] = color
        else:
            if normalize:
                mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_mask = cv2.applyColorMap(mask, colormap)
            
        assert len(colored_mask.shape) == 3, "Output mask must be a 3D array"
        return colored_mask

    @staticmethod
    def get_random_crop(img, size, m=None):
        assert isinstance(img, np.ndarray), "Input image must be a numpy array"
        assert isinstance(m, np.ndarray), "Input array must be numpy array"
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        if m is not None:
            assert img.shape[:2] == m.shape[:2], "Image and mask must have the same size"
            m = m[y : y + size, x : x + size]
        img = img[y : y + size, x : x + size]
        return img, m

    @staticmethod
    def get_central_crop(img, size):
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        half_size = size // 2
        img = img[center_y - half_size: center_y + half_size, center_x - half_size: center_x + half_size]
        return img

    def __getitem__(self, idx):   
        pass
    
    @staticmethod
    def get_edges(t):
        edge = np.zeros_like(t, dtype=bool)
        edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
        edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])

        return edge.astype(float)
 
    @staticmethod
    def resize_with_aspect_ratio(tile, target_size):
        if isinstance(tile, np.ndarray):
            tile = Image.fromarray(tile)
        w, h = tile.size
        if h < target_size or w < target_size:
            aspect_ratio = w / h
            if w > h:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            else:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            tile = tile.resize((new_w, new_h), Image.BICUBIC)
        if isinstance(tile, Image.Image):
            return np.array(tile)
        return tile

class PanNukeDataset(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.indices = list(range(len(self.data['images'])))
        self.inference = config.get("inference", False)

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
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # We need to change the mask to have it the numbers that are evenly distributed in the range of 0 to 255.
        # map_mask = np.vectorize(self.map_values)
        # mask = map_mask(mask, t='identity')   
        mask = mask.astype(np.uint8)   
        mask = self.apply_colormap_to_mask(mask, FIXED_COLOR_MAP) # normalize and convert to RGB.   
        
        if np.random.rand() < self.p_uncond:
            mask = np.zeros(mask.shape, dtype=np.uint8)
            # mask = np.zeros(mask.shape[:-1], dtype=np.uint8) 
            # mask = self.apply_colormap_to_mask(mask, normalize=True) # applying colormap to the null mask too 
            
        return {
            "image": processed_image,
            "mask": mask,
            "caption": "",
            "type": type
        }

    def __len__(self):
        return len(self.indices)

# this version refelcts the newest split of the Pannuke dataset that is consistent with Evaluation
class PanNukeDatasetV2(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.inference = config.get("inference", False)
        if self.inference: # at inference time we want fixed order of the files
            self.data['fnames'] = sorted(self.data['fnames'])
            seed=config.get("seed", 42)
            random.Random(seed).shuffle(self.data['fnames'])
        self.hint_channels = config.get("hint_channels", 3)
        self.class_names_to_labels = {'Neoplastic': 1, 'Inflammatory': 2, 'Connective': 3, 'Dead': 4, 'Epithelial': 5}
        
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
        inst_mask = self.get_edges(mask['inst_map'])
        type_mask = mask['type_map']
        tissue_type = self.data['types'].loc[fname+".png", 'type']
        cell_counts = self.data['cell_counts'].loc[fname+".png", :].values

        if self.crop_size:
            type_mask = self.get_central_crop(type_mask, self.crop_size)
            inst_mask = self.get_central_crop(inst_mask, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32)
        type_mask = np.array(type_mask, dtype=np.uint8)
        inst_mask = np.array(inst_mask, dtype=np.uint8)
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            type_mask = np.flip(type_mask, axis=0).copy()
            inst_mask = np.flip(inst_mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            type_mask = np.flip(type_mask, axis=1).copy()
            inst_mask = np.flip(inst_mask, axis=1).copy()
        
        # We need to change the mask to have it the numbers that are evenly distributed in the range of 0 to 255.
        type_mask = type_mask.astype(np.uint8)   
        type_mask = self.apply_colormap_to_mask(type_mask, FIXED_COLOR_MAP) # fixed colormap 
        inst_mask = inst_mask.astype(np.uint8)
        inst_mask[inst_mask==1] = 9 # reserved for edges, it will be shown as white. 
        inst_mask = self.apply_colormap_to_mask(inst_mask, FIXED_COLOR_MAP) 
        
        # save_image(processed_image, f"processed_image{fname}.jpg")
        # save_image(inst_mask, f"inst_mask{fname}.jpg")  
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            type_mask = np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)
            inst_mask = np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)
        
        if self.hint_channels == 6:  
            cond_mask = np.concatenate([type_mask, inst_mask], axis=2)
        else:
            assert self.hint_channels == 3, "Hint channels must be 3 or 6"
            cond_mask = type_mask
        ret = {
                "image": processed_image,
                "mask": cond_mask,
                "caption": "",
                "type": tissue_type,
                "cell_counts": cell_counts,
                "fname": fname
            }
        return ret

    def __len__(self):
        return len(self.data['fnames'])

class SegPathDataset(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.inference = config.get("inference", False)
        self.class_names_to_labels = {'SmoothMuscle':1, 'RBC':2, 'Lymphocyte':3, 'PlasmaCell':4, 'Epithelium':5} 
        
    def get_core_filename(self, filename):
        core_filename = '_'.join(filename.split('_')[:-1])
        return core_filename
        
    def read_data(self, data_dir, split):
        data = {'fnames': []}
        self.data_dir = data_dir
        self.antibodies = ['aSMA', 'CD235a', 'CD3CD20', 'MIST1', 'panCK']
        # self.anitbodies = [self.antibodies[4]]
        for abody in self.antibodies:
            csv_path = os.path.join(self.data_dir, f'{abody}_fileinfo.csv')
            df = pd.read_csv(csv_path) 
            if split == 'train':
                df = df[(df['train_val_test'] == 'train') | (df['train_val_test'] == 'val')]
            else:
                df = df[(df['train_val_test'] == 'test')]
            for idx, row in df.iterrows():
                data['fnames'].append(self.get_core_filename(row['filename']))
        random.shuffle(data['fnames'])
        return data
    
    def __getitem__(self, idx):
        fname = self.data["fnames"][idx]
        image = cv2.imread(os.path.join(self.data_dir, fname+"_HE.png"))
        type_mask = cv2.imread(os.path.join(self.data_dir, fname+"_mask.png"), cv2.IMREAD_GRAYSCALE)

        # save_image(image, f"image0.jpg")
        # save_image(type_mask, f"type_mask0.jpg")
        if self.crop_size:
            image, type_mask = self.get_random_crop(image, self.crop_size,  type_mask)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32) 
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            type_mask = np.flip(type_mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            type_mask = np.flip(type_mask, axis=1).copy()
        
        cell_type = fname.split('_')[2]
        type_mask[type_mask==1] = self.class_names_to_labels[cell_type]
        type_mask = self.apply_colormap_to_mask(type_mask, FIXED_COLOR_MAP)   
        
        # save_image(processed_image, f"processed_image1.jpg", normalize=True)
        # save_image(type_mask, f"type_mask1.jpg", normalize=False)  
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            type_mask = np.ones_like(type_mask, np.uint8)*NULL_MASK
        ret = {
                "image": processed_image,
                "mask": type_mask,
                "caption": "",
                "fname": fname
            }
        return ret

    def __len__(self):
        return len(self.data['fnames'])
    

class ConicDataset(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.inference = config.get("inference", False)
        self.class_names_to_labels = { 'neutrophil': 1, 'epithelial': 2, 'lymphocyte': 3,
                                      'plasma': 4, 'eosinophil': 5, 'connective' : 6} 
        self.hint_channels = config.get("hint_channels", 3)
        
    def read_data(self, data_dir, split):
        data = {}
        meta_data = pd.read_csv(os.path.join(data_dir, 'counts.csv')) 
        data['meta_data'] = meta_data[meta_data['split'] == split]
        data['images'] = np.load(os.path.join(data_dir, 'images.npy'))
        data['labels'] = np.load(os.path.join(data_dir, 'labels.npy'))
        self.split = split
        return data
    
    def __getitem__(self, idx):
        meta_data = self.data['meta_data'].iloc[idx]
        data_idx = meta_data.name
        image = self.data['images'][data_idx]
        inst_mask = self.get_edges(self.data['labels'][data_idx, :, :, 0])
        type_mask = self.data['labels'][data_idx, :, :, 1]
        tissue_type = "human_organ" # we dont know the tissue type
        cell_counts = list(meta_data[:-1].values) # skip the split

        if self.crop_size:
            type_mask = self.get_central_crop(type_mask, self.crop_size)
            inst_mask = self.get_central_crop(inst_mask, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32)
        type_mask = np.array(type_mask, dtype=np.uint8)
        inst_mask = np.array(inst_mask, dtype=np.uint8)
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            type_mask = np.flip(type_mask, axis=0).copy()
            inst_mask = np.flip(inst_mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            type_mask = np.flip(type_mask, axis=1).copy()
            inst_mask = np.flip(inst_mask, axis=1).copy()
        type_mask = type_mask.astype(np.uint8)   
        type_mask = self.apply_colormap_to_mask(type_mask, FIXED_COLOR_MAP) # fixed colormap 
        inst_mask = inst_mask.astype(np.uint8)
        inst_mask[inst_mask==1] = 9 # reserved for edges, it will be shown as white. 
        inst_mask = self.apply_colormap_to_mask(inst_mask, FIXED_COLOR_MAP) 
        
        # save_image(processed_image, f"processed_image{fname}.jpg")
        # save_image(inst_mask, f"inst_mask{fname}.jpg")  
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            type_mask = np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)
            inst_mask = np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)
        
        if self.hint_channels == 6:  
            cond_mask = np.concatenate([type_mask, inst_mask], axis=2)
        else:
            assert self.hint_channels == 3, "Hint channels must be 3 or 6"
            cond_mask = type_mask
        ret = {
                "image": processed_image,
                "mask": cond_mask,
                "caption": "",
                "type": tissue_type,
                "cell_counts": cell_counts,
                "fname": data_idx
            }
        # print([type(ret[k]) for k in ret.keys()])
        return ret
    def __len__(self):
        return self.data['meta_data'].shape[0]



class MonuSacDataset(MaskedDataset):
    def __init__(self, config=None):
        super().__init__(config)
        self.inference = config.get("inference", False)
        self.class_names_to_labels = {  'Epithelial': 1, 'Lymphocyte': 2, 'Macrophage': 4,
                                      'Neutrophil': 3, 'Ambiguous': 5} 
        self.hint_channels = config.get("hint_channels", 3)
        assert self.hint_channels == 3, "Hint channels must be three, no instance map present for this dataset"
        
    def read_data(self, data_dir, split):
        data = {}
        self.data_dir = data_dir
        data['meta_data'] = pd.read_csv(os.path.join(data_dir, f'monusac_{split}', f'{split}_meta.csv')) 
        self.split = split
        return data
    
    def get_image_tile_path(self, mask_tile_path):
        dir_path, mask_filename = os.path.split(mask_tile_path)
        image_filename = mask_filename.replace('mask_tile', 'image_tile').replace('.npy', '.png')
        image_tile_path = os.path.join(dir_path, image_filename)
        return image_tile_path

    def __getitem__(self, idx):
        meta_data = self.data['meta_data'].iloc[idx]
        image_fname = os.path.join(self.data_dir, f'monusac_{self.split}', self.get_image_tile_path(meta_data['filename']))
        image = cv2.imread(image_fname)
        image = self.resize_with_aspect_ratio(image, self.crop_size+1)  
        type_mask = np.load(os.path.join(self.data_dir, f'monusac_{self.split}', meta_data['filename'].split('.')[0]+".npy"))
        type_mask = self.resize_with_aspect_ratio(type_mask, self.crop_size+1)
        tissue_type = "human_organ"
        cell_counts = list(meta_data[1:].values) # skip the filename
        assert len(image.shape)==3 and image.shape[-1] == 3, "Image must be a 3 channel image"

        if self.crop_size:
            type_mask = self.get_central_crop(type_mask, self.crop_size)
            image = self.get_central_crop(image, self.crop_size)
        
        processed_image = (image / 127.5 - 1.0).astype(np.float32)
        type_mask = np.array(type_mask, dtype=np.uint8)
        # Random horizontal and vertical flips
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=0).copy()
            type_mask = np.flip(type_mask, axis=0).copy()
        if np.random.rand() < 0.5 and not self.inference:
            processed_image = np.flip(processed_image, axis=1).copy()
            type_mask = np.flip(type_mask, axis=1).copy()
        
        type_mask = type_mask.astype(np.uint8)   
        type_mask = self.apply_colormap_to_mask(type_mask, FIXED_COLOR_MAP) # fixed colormap 
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            type_mask = np.ones_like(type_mask, np.uint8)*NULL_MASK
        cond_mask = type_mask
        
        parts = image_fname.split("/")
        base_name = parts[-3]  # Gets patient
        subfolder = parts[-2]  # Gets patient subdirectory
        file_name = os.path.splitext(parts[-1])[0]  # Gets tile
        ret_fname = f"{base_name}_{subfolder}_{file_name}"
        ret = {
                "image": processed_image,
                "mask": cond_mask,
                "caption": "",
                "type": tissue_type,
                "cell_counts": cell_counts,
                "fname": ret_fname
            }
        return ret
    
    def __len__(self):
        return self.data['meta_data'].shape[0]