from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import io


class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        split = config.get("split")
        self.split = split
        data_dir = Path(config.get("root"))
        self.crop_size = config.get("crop_size", None)

        num_levels = config.get("num_levels", 2)
        assert num_levels in [2, 3, 4], "num_levels must be 2 or 3 or 4"
        self.p_uncond = config.get("p_uncond", 0)

        # Low, high if two levels else low, mid, high
        self.levels = ["low", "high"] if num_levels == 2 else ["low", "mid", "high"]

        # Load .h5 dataset
        self.data_file = h5py.File(data_dir / "TCGA_BRCA_10x_448_tumor.hdf5", "r")

        # Load metadata
        arr1 = np.load(data_dir / f"train_test_brca_tumor/{split}.npz", allow_pickle=True)
        self.indices = arr1["indices"]
        self.summaries = arr1["summaries"].tolist()
        self.prob_tumor = arr1["prob_tumor"].tolist()
        self.prob_til = arr1["prob_til"].tolist()

    def __len__(self):
        # if self.split == 'train': # TODO Mahesh: comment this line, only for debugging and running on small dataset
        #     return 500
        # else:
        #     return 10
        return len(self.indices)

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
        x_idx = self.indices[idx]

        tile = self.data_file["X"][x_idx]
        tile = np.array(Image.open(io.BytesIO(tile)))

        image = (tile / 127.5 - 1.0).astype(np.float32)
        if self.crop_size:
            image = self.get_random_crop(image, self.crop_size)
            # image = self.get_central_crop(image, self.crop_size)

        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()

        wsi_name = self.data_file["wsi"][x_idx].decode()
        folder_name = self.data_file["folder_name"][x_idx].decode()
        text_prompt = self.summaries[wsi_name]

        # Convert tumor infiltrating lymphocytes to levels low / mid / high and add to text prompt
        p_til = self.prob_til.get(wsi_name, {}).get(folder_name)
        if p_til is not None:
            p_til = int(p_til * len(self.levels))
            text_prompt = f"{self.levels[p_til]} til; {text_prompt}"

        # Convert tumor presence to levels low / mid / high and add to text prompt
        p_tumor = self.prob_tumor.get(wsi_name, {}).get(folder_name)
        if p_tumor is not None:
            p_tumor = int(p_tumor * len(self.levels))
            text_prompt = f"{self.levels[p_tumor]} tumor; {text_prompt}"

        # Replace text prompt with unconditional text prompt with probability p_uncond
        # Dont replace if p_til is positive
        if np.random.rand() < self.p_uncond and (p_til is None or p_til == 0):
            text_prompt = ""

        return {
            "image": image,
            "caption": text_prompt,
        }
