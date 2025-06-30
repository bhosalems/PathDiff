import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, config):
        self.data = []
        self.split = config.split
        self.p_uncond = config.p_uncond
        self.root_dir = config.root_dir + self.split + "_split/"
        with open(self.root_dir + '/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(self.root_dir + source_filename)
        target = cv2.imread(self.root_dir + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        if np.random.rand() < self.p_uncond:
            prompt = ""
        
        if self.split == "validation": # forcing validation to have null mask for some tests, #FIXME
            if np.random.rand() < 0.9:
                source = np.zeros_like(source, np.uint8)
        
        if np.random.rand() < self.p_uncond: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
            source = np.zeros_like(source, np.uint8)

        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == "__main__":
    dataset = MyDataset()
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)