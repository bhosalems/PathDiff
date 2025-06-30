import os
import einops
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ldm.data.mask_cond.mask_condition import FIXED_COLOR_MAP, NULL_MASK
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
import torch
import cv2
import numpy as np
from ldm.util import instantiate_from_config
import random
import pandas as pd
from tqdm import tqdm

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
    9: (255, 255, 255)     # Cell type 9 - White (For edges)
}

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

def get_edges(t):
    edge = np.zeros_like(t, dtype=bool)
    edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
    edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
    edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])

    return edge.astype(float)

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    sd = pl_sd["state_dict"]
    config.model.params.cond_stage_config.params["device"] = device
    model = instantiate_from_config(config.model) # do not use resume path
    model.resume_path = None
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def get_model(config_path,  device, checkpoint):
    config = OmegaConf.load(config_path)
    if 'ckpt_path' in config['model']['params']['first_stage_config']['params'].keys():
        del config['model']['params']['first_stage_config']['params']['ckpt_path'] 
    
    if 'ckpt_path' in config['model']['params']['unet_config']['params'].keys():
        del config['model']['params']['unet_config']['params']['ckpt_path']
    model = load_model_from_config(config, checkpoint, device)
    return model

def get_unconditional_token(batch_size):
    return [""]*batch_size

def log_txt_as_img(wh, xc, initial_size=16, subplot_scale=1.0):
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        
        size = initial_size
        while size > 5:  # Minimum font size to prevent infinite loop
            try:
                font = ImageFont.truetype("data/DejaVuSans.ttf", size=int(size * subplot_scale))
            except OSError:
                font = ImageFont.load_default()

            max_width = int(wh[0] * 0.8)  # 80% of the image width
            threshold = max_width // int(draw.textbbox((0, 0), 'A', font=font)[2] * subplot_scale)
            lines = "\n".join(xc[bi][start : start + threshold] for start in range(0, len(xc[bi]), threshold))
            bbox = draw.textbbox((10, 10), lines, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if text_height <= wh[1] * 0.9:
                break
            size -= 2
        
        draw.text((10, 10), lines, fill="black", font=font)
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def add_edge_type_mask(control, change_color=False, err=False):
    # Check if control is a PyTorch tensor or a NumPy array and handle accordingly
    is_tensor = isinstance(control, torch.Tensor)
    
    if control.shape[-1] != 6:
        if err:
            assert True, "Control tensor should have 6 channels (3 for type and 3 for instance)."
        else:
            UserWarning("Control tensor should have 6 channels (3 for type and 3 for instance).")
            return control
    
    if is_tensor:
        type_mask = control[..., :3].clone()  # Extract type mask with shape (1, 256, 256, 3)
        inst_mask = control[..., 3:].clone()  # Extract instance mask with shape (1, 256, 256, 3)
    else:
        type_mask = control[..., :3].copy()  # For NumPy arrays, use .copy() instead of .clone()
        inst_mask = control[..., 3:].copy()

    if change_color:
        combined_mask = type_mask + inst_mask
        min_val = combined_mask.min(axis=-1, keepdims=True)
        max_val = combined_mask.max(axis=-1, keepdims=True)
        combined_mask = (combined_mask - min_val) / (max_val - min_val + 1e-8)  # Add small value to prevent division by zero
        return combined_mask
    else:
        edge_positions = (inst_mask > 0)  # Identify edge positions
        
        if is_tensor:
            # Modify type mask at edge positions for PyTorch tensor
            type_mask[..., 0][edge_positions[..., 0]] = 255  # Red channel
            type_mask[..., 1][edge_positions[..., 1]] = 255  # Green channel
            type_mask[..., 2][edge_positions[..., 2]] = 255  # Blue channel
        else:
            # Modify type mask at edge positions for NumPy array
            type_mask[..., 0][edge_positions[..., 0]] = 255  # Red channel
            type_mask[..., 1][edge_positions[..., 1]] = 255  # Green channel
            type_mask[..., 2][edge_positions[..., 2]] = 255  # Blue channel
        return type_mask

def get_image_files(root, extensions=('.jpg', '.jpeg', '.png')):
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_files.append(os.path.join(dirpath, filename))
    return sorted(image_files)

def sample_folder(model, data, output_dir, dataset="TCGA", mask_channels=3, save=True, grid=False, number_of_examples=10):
    assert mask_channels in [3, 6]
    text_dir = os.path.join(output_dir, "text_images")
    mask_dir = os.path.join(output_dir, "mask_images")
    generated_dir = os.path.join(output_dir, "generated_images")
    if dataset == "TCGA":
        captions = pd.read_csv("/a2il/data/mbhosale/PathDiff/PathCap/wrap/pathcap_label_pred.csv")
    elif dataset == "PATHCAP":
        captions = pd.read_csv(os.path.join("/".join(data.split("/")[:-1]), "label_pred_annots.csv"))
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    generated_images = []
    text_images = []
    mask_images = []
    i = 0
    # input_files = get_image_files(os.path.join(data, "masks"), extensions=('.npy'))
    input_files = ['Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma/0@16336672_1742-6413-2-21-3.npy',
                   'Uveal_Melanoma/0@17249508_kjim-21-248-g002_1.npy',
                   'Esophageal_carcinoma/0@20224681_SRCM2010-907127.npy',
                   'Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma/0@22276235_SNI-2-181-g003_1.npy',
                   'Esophageal_carcinoma/0@23342295_arh0021200350002.npy',
                   'Breast_invasive_carcinoma/0@23378963_NAJMS-5-72-g003.npy',
                   'Esophageal_carcinoma/0@25045357_CRIM2014-952038.npy',
                   'Liver_hepatocellular_carcinoma/0@27110418_CRIPA2016-1738696.npy',
                   'Lung_adenocarcinoma/0@27222791_gr2.npy',
                   'Lung_adenocarcinoma/0@28740441_lctt-8-045Fig9_0.npy',
                   'Breast_invasive_carcinoma/0@32089923_CRIONM2020-6245415.npy',
                   'Kidney_renal_papillary_cell_carcinoma/0@33665128_gr2.npy',
                   'Lung_adenocarcinoma/0@34434479_jmc-12-310-g002_0.npy',
                   'Breast_invasive_carcinoma/0@34621839_WJCC-9-7863-g001_1.npy'
                   ]
    input_files = [os.path.join(data, "masks", f) for f in input_files]
    for mask_path in tqdm(input_files):
        if i >= number_of_examples:
            break
        cls = None
        if dataset == 'CONIC':
            mask = np.load(mask_path, allow_pickle=True)
            # inst_mask = get_edges(mask[:, :, 0])
            inst_mask = None
            type_mask = mask[:, :, 1]
            caption = random.choice(["Lung squamous cell carcinoma", "Colon adenocarcinoma", "Lung adenocarcinoma"])
        elif dataset == 'TCGA':
            type_mask = np.load(mask_path)
            inst_mask = None
            # ins_mask = get_edges(type_mask)
            if contains_case_insensitive(mask_path, "colon_adenocarcinoma"):
                colon_keywords = ['colon', 'colonic', 'colorectal', 'large intestine', 'cecum']
                captions = captions[captions['split'] == 'train']
                pattern = r'\b(?:' + '|'.join(colon_keywords) + r')\b'
                colon_text = captions[(captions['cancer_type'] == 'adenocarcinoma') & 
                                      (captions['text'].str.contains(pattern, case=False, regex=True))]['text']
                caption = random.choice(colon_text.to_list())
                cls = "Colon_adenocarcinoma"
            elif contains_case_insensitive(mask_path, "lung_adenocarcinoma"):
                lung_adeno_keywords = ['lung', 'pulmonary', 'bronchial']
                lung_adeno_pattern = r'\b(?:' + '|'.join(lung_adeno_keywords) + r')\b'
                lung_adeno_text = captions[(captions['cancer_type'] == 'adenocarcinoma') & 
                                        (captions['text'].str.contains(lung_adeno_pattern, case=False, regex=True))]['text']
                caption = random.choice(lung_adeno_text.to_list())
                cls = "Lung_adenocarcinoma"
            elif contains_case_insensitive(mask_path, "lung_squamous_cell_carcinoma"):
                lung_scc_keywords = ['lung', 'pulmonary', 'bronchial']
                lung_scc_pattern = r'\b(?:' + '|'.join(lung_scc_keywords) + r')\b'
                lung_scc_text = captions[(captions['cancer_type'] == 'squamous') & 
                                        (captions['text'].str.contains(lung_scc_pattern, case=False, regex=True))]['text']
                caption = random.choice(lung_scc_text.to_list())
                cls = "Lung_squamous_cell_carcinoma"
            else:
                assert NotImplementedError("Not implemented cancer type")
        elif dataset == 'PATHCAP':
            type_mask = np.load(mask_path)
            if mask_channels == 6:
                inst_mask = get_edges(type_mask)
            captions_i = captions[captions['img'].str.contains(mask_path.split("/")[-1].split(".")[0])] # select caption for the given image
            caption = captions_i['text'].values[0]
            cls = captions_i['cancer_type'].values[0]
        else:
            mask = np.load(mask_path, allow_pickle=True).item()
            inst_mask = get_edges(mask['inst_map'])
            type_mask = mask['type_map']
            caption = random.choice(["Lung Squamos Cell Carcinoma", "Colon adeocarcinoma", "Lung adenocarcinoma"])
        generated_image, mask_image, text_image = sample_one(model, type_mask, caption, inst_mask=inst_mask)
        generated_images.append(generated_image)
        text_images.append(text_image)
        mask_images.append(mask_image)
        
        if cls is None:
            if contains_case_insensitive(caption, "colon adenocarcinoma"):
                    cls = "Colon_adenocarcinoma"
            elif contains_case_insensitive(caption, "lung adenocarcinoma"):
                    cls = "Lung_adenocarcinoma"
            elif contains_case_insensitive(caption, "lung squamous cell carcinoma"):
                    cls = "Lung_squamous_cell_carcinoma"
        if save:
            fname = mask_path.split("/")[-1].split(".")[0]
            generated_cls_dir = os.path.join(generated_dir, cls)
            os.makedirs(generated_cls_dir, exist_ok=True)
            mask_cls_dir = os.path.join(mask_dir, cls)
            os.makedirs(mask_cls_dir, exist_ok=True)
            text_cls_dir = os.path.join(text_dir, cls)
            os.makedirs(text_cls_dir, exist_ok=True)
            Image.fromarray(generated_image).save(os.path.join( generated_cls_dir, fname+".png"))
            Image.fromarray(mask_image).save(os.path.join(mask_cls_dir, fname+".png"))
            Image.fromarray(text_image).save(os.path.join(text_cls_dir, fname+".png"))
        i+=1
        print(f"generated [{i}/{number_of_examples}]")
    
    print(f"Images saved in {output_dir}")
    if grid:
        # Display the results in a 2x4 grid for each set of generated image, mask, and text
        fig, axes = plt.subplots(3, 8, figsize=(24, 12))  # 3 rows: text, mask, and generated image
        for i in range(number_of_examples):
            axes[0, i].imshow(text_images[i])
            axes[0, i].set_title("Text")
            axes[0, i].axis("off")

            axes[1, i].imshow(mask_images[i])
            axes[1, i].set_title("Mask")
            axes[1, i].axis("off")

            axes[2, i].imshow(generated_images[i])
            axes[2, i].set_title("Generated Image")
            axes[2, i].axis("off")

        # Save the subplot as a file instead of showing it interactively
        save_path = os.path.join(output_dir, "results_in_grid.png")
        plt.savefig(save_path, dpi=300)  # Adjust dpi if needed
        plt.close()  

def contains_case_insensitive(s, sub):
    return sub.lower() in s.lower()
       
def sample_one(model, type_mask, caption, inst_mask=None, number_of_steps = 200, unconditional_guidance_scale = 1.75, mode=''):
    type_mask = np.array(type_mask, dtype=np.uint8)
    type_mask = apply_colormap_to_mask(type_mask, FIXED_COLOR_MAP)
    type_mask = torch.from_numpy(type_mask).unsqueeze(0)
    
    if inst_mask is not None:
        inst_mask = np.array(inst_mask, dtype=np.uint8)
        inst_mask[inst_mask == 1] = 9
        inst_mask = apply_colormap_to_mask(inst_mask, FIXED_COLOR_MAP)
        inst_mask = torch.from_numpy(inst_mask).unsqueeze(0)
        mask=torch.from_numpy(np.concatenate([type_mask, inst_mask], axis=3)).to(device)
    else:
        mask = type_mask.to(device)
        
    # Conditions and parameters
    batch_size = 1
    if mode == 't2i':
        conds = {"caption":caption, "mask": torch.from_numpy(np.full((256, 256, 6), NULL_MASK, dtype=np.uint8)).unsqueeze(0).to(device)}
    elif mode == 'm2i':
        conds = {"caption": "", "mask": mask}
    else:
        conds = {"caption": caption, "mask": mask}

    with torch.no_grad():
        ut = get_unconditional_token(batch_size)
        uc = model.get_learned_conditioning(ut)
        ct = conds["caption"]
        cc = model.get_learned_conditioning(ct)
        if hasattr(model, 'control_key'):
            control = conds["mask"]
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float().to(device)
            cc = dict(c_crossattn=[cc], c_concat=[control])
            uc = dict(c_crossattn=[uc], c_concat=[control])
        
        samples_ddim, _ = sampler.sample(
            number_of_steps, batch_size, [3, 64, 64], cc, verbose=False,
            unconditional_conditioning=uc, unconditional_guidance_scale=unconditional_guidance_scale, eta=1
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()
        
        generated_image = x_samples_ddim.squeeze(0).permute(1, 2, 0).numpy()
        mask_image = np.squeeze(add_edge_type_mask(control.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)))
        
        text_image = log_txt_as_img((256, 256), [ct], initial_size=16)[0].permute(1, 2, 0).numpy()
        text_image = ((text_image + 1) * 127.5).astype(np.uint8)
        
        # Convert to numpy array and add to lists
        return generated_image, mask_image, text_image

if __name__ == "__main__":
    dataset='PATHCAP'
    device = "cuda"
    number_of_examples = 500
    method = 'PathDiff'  
    data =  "pathcap_label_pred_small/annots"
    output_dir = "samples"
    if method == 'PathDiff':
        mask_channels = 6
        model_check_points = {
            "PanNuke": "",
            "TCGA": "",
            "CONIC": "",
            "PATHCAP": "last.ckpt"
        }
        model_configs = {
            "PanNuke": "",
            "TCGA": "",
            "CONIC": "",
            "PATHCAP": "/configs/11-02T02-36-project.yaml"
        }
    elif method == 'ControlNet':
        if dataset in ['CONIC', 'PATHCAP']:
            mask_channels = 6
        model_check_points = {
            "CONIC": "",
            "PATHCAP": ""
        }
        model_configs = {
            "CONIC": "",
            "PATHCAP": ""
        }  
    model = get_model(model_configs[dataset], device, model_check_points[dataset])
    sampler = DDIMSampler(model)
    sample_folder(model, data=data, output_dir=output_dir, number_of_examples=number_of_examples, dataset=dataset, mask_channels=mask_channels)