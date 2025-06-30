import os
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
NULL_MASK = 100
class MixedCoCo(Dataset):
    """
    Dataset for disjoint Mask->Image (M2I) and Text->Image (T2I) pairs,
    where T2I captions must not mention any of the selected mask classes.
    """
    def __init__(
        self,
        config
    ):
        root = config.get('root')
        self.mode = config.get("mode")
        assert self.mode in ('M2I','T2I', 'MIXED')
        self.p_uncond = 0.05 #config.get("p_uncond")
        self.mask_classes = set(config.get("mask_classes") or [])
        self.crop_size = int(config.get("crop_size"))
        self.inference = config.get("inference")
        self.img_folder = os.path.join(root, "images", config.get("split")+"2017")
        self.semantic_mask_dir = os.path.join(root, "images", config.get("split")+"2017_thing_semantic_masks")
        self.split = config.get("split")
        self.p_split = config.get("p_split", 0.5)

        # initialize COCO APIs
        self.coco_mask = COCO(os.path.join(root, os.path.join(root, "annotations", "instances_"+self.split+"2017.json")))
        self.coco_cap  = COCO(os.path.join(root, os.path.join(root, "annotations", "captions_"+self.split+"2017.json") ))

        # map mask class IDs to their category names
        mask_cats = self.coco_mask.loadCats(list(self.mask_classes))
        self.mask_cat_names = {c['name'].lower() for c in mask_cats}
        
        # split image IDs
        all_ids = set(self.coco_mask.getImgIds())
        m2i_ids, t2i_ids = [], []

        for img_id in all_ids:
            # check segmentation classes
            ann_ids = self.coco_mask.getAnnIds(imgIds=[img_id], iscrowd=None)
            cats    = {a['category_id'] for a in self.coco_mask.loadAnns(ann_ids)}

            if cats & self.mask_classes:
                m2i_ids.append(img_id)
            else:
                # ensure captions exist
                cap_ids = self.coco_cap.getAnnIds(imgIds=[img_id])
                if not cap_ids:
                    continue

                # load captions and filter out any that mention mask categories
                anns     = self.coco_cap.loadAnns(cap_ids)
                captions = [ann['caption'].lower() for ann in anns]
                # skip if any caption contains a mask category name
                if any(any(name in cap for name in self.mask_cat_names) for cap in captions):
                    continue

                t2i_ids.append(img_id)

        self.m2i_ids = m2i_ids
        self.t2i_ids = t2i_ids
        self.m2i_length = len(m2i_ids)
        self.t2i_length = len(t2i_ids)
        # if self.mode == 'MIXED' or self.mode == 'M2I':
        #     self.img_key = "jpg"
        #     self.mask_key = "hint"
        #     self.txt_key = "txt"
        # elif self.mode == 'T2I':
        #     self.img_key = "image"
        #     self.mask_key = "caption"

    def __len__(self):
        if self.mode == 'M2I':
            return self.m2i_length
        elif self.mode == 'T2I':
            return self.t2i_length
        elif self.mode == 'MIXED':
            return self.m2i_length + self.t2i_length
            # return min(len(self.m2i_ids), len(self.t2i_ids))
    
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
    
    @staticmethod
    def get_central_crop(img, size):
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        half_size = size // 2
        img = img[center_y - half_size: center_y + half_size, center_x - half_size: center_x + half_size]
        return img
    
    def __getitem__(self, idx):
        mode = self.mode
        if self.mode == "MIXED":
            if np.random.rand() < self.p_split:
                mode = 'M2I'
                idx = idx % self.m2i_length
            else:
                mode = 'T2I'
                idx = idx % self.t2i_length
        if mode == 'M2I':
            img_id = self.m2i_ids[idx]
        elif mode == 'T2I':
            img_id = self.t2i_ids[idx]
        # info   = self.coco_mask.loadImgs(img_id)[0]
        path = os.path.join(self.coco_mask.imgs[img_id]['file_name'])
        tile = Image.open(os.path.join(self.img_folder, path))
        if tile.mode != 'RGB':
            tile = tile.convert("RGB")
        # tile = self.resize_with_aspect_ratio(tile, self.crop_size+1)
        tile = tile.resize((self.crop_size, self.crop_size), resample=Image.LANCZOS)
        tile = np.array(tile)
        image = (tile / 127.5 - 1.0).astype(np.float32)
        # if self.crop_size:
            # image = self.get_random_crop(image, self.crop_size)
            # image = self.get_central_crop(image, self.crop_size)
        # if self.inference:
        #     idx = idx % self.m2i_length
        #     img_id = self.m2i_ids[idx]
        #     path = os.path.join(self.coco_mask.imgs[img_id]['file_name'])
        #     tile = Image.open(os.path.join(self.img_folder, path))
        #     if tile.mode != 'RGB':
        #         tile = tile.convert("RGB")
        #     # tile = self.resize_with_aspect_ratio(tile, self.crop_size+1)
        #     tile = tile.resize((self.crop_size, self.crop_size), resample=Image.LANCZOS)
        #     tile = np.array(tile)
        #     image = (tile / 127.5 - 1.0).astype(np.float32)
            
        #     mask = Image.open(os.path.join(self.semantic_mask_dir, "".join(path.split(".")[0])+".png"))
        #     mask = mask.resize((self.crop_size, self.crop_size), resample=Image.NEAREST)
        #     mask = np.array(mask, dtype=np.uint8)

        #     # mask = self.resize_with_aspect_ratio(mask, self.crop_size+1)
        #     # if self.crop_size:
        #         # mask  = self.get_central_crop(mask, self.crop_size)
        #     cap_ids = self.coco_cap.getAnnIds(imgIds=[img_id])
        #     anns = self.coco_cap.loadAnns(cap_ids)
        #     caption = random.choice(anns)['caption']
        #     return {"jpg": image, "txt": caption, 'hint': mask}
        #         # return {"image": image, "caption": caption}
        # else:
        if mode == 'M2I':
            mask = Image.open(os.path.join(self.semantic_mask_dir, "".join(path.split(".")[0])+".png"))
            mask = mask.resize((self.crop_size, self.crop_size), resample=Image.NEAREST)
            mask = np.array(mask, dtype=np.uint8)
            # mask = self.resize_with_aspect_ratio(mask, self.crop_size+1)
            # if self.crop_size:
                # mask  = self.get_central_crop(mask, self.crop_size)
            if np.random.rand() < 0.5 and not self.inference:
                image = np.flip(image, axis=0).copy()
                mask  = np.flip(mask, axis=0).copy()
            if np.random.rand() < 0.5 and not self.inference:
                image = np.flip(image, axis=1).copy()
                mask  = np.flip(mask, axis=1).copy()    
            if np.random.rand() < self.p_uncond and not self.inference: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step().
                mask = np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)
            return {"jpg": image, "txt": "", 'hint': mask}
            
        else:  # T2I
            if np.random.rand() < 0.5 and not self.inference:
                image = np.flip(image, axis=0).copy()
            if np.random.rand() < 0.5 and not self.inference:
                image = np.flip(image, axis=1).copy()  
            cap_ids = self.coco_cap.getAnnIds(imgIds=[img_id])
            anns    = self.coco_cap.loadAnns(cap_ids)
            caption = random.choice(anns)['caption']
            if np.random.rand() < self.p_uncond and not self.inference: # effectively same as UCG training in ddpm_control.py -> DDPM -> training_step()
                caption = ""
            return {"jpg": image, "txt": caption, 'hint': np.full((256, 256, 3), NULL_MASK, dtype=np.uint8)}
            
# if __name__ == "__main__":
#     import torchvision.transforms as T
#     from torch.utils.data import DataLoader

#     # Example usage
#     dataset = MixedCoCo(
#         root='/a2il/data/mbhosale/coco/',
#         # ann_mask='annotations/instances_train2017.json',
#         # ann_cap='annotations/captions_train2017.json',
#         mask_classes=[16, 22, 23, 24, 25]
#         # example class IDs
#         mode='T2I',
#     )

#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for imgs, masks in dataloader:
#         print(imgs.shape, masks.shape)