import warnings
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
import math
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
warnings.filterwarnings("ignore")
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm
import os
import argparse
from torchvision.transforms.functional import to_pil_image
import einops
import timm
from transformers import CLIPProcessor, CLIPModel
from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from main import add_edge_type_mask
from torch.utils.data import DataLoader
from ldm.data.mask_cond.mask_condition import NULL_MASK

# device = torch.device('cuda:0')
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

def get_unconditional_mask(size):
    return torch.zeros(size, dtype=torch.uint8)
    
def get_conditional_token(batch_size, summary):

    # append tumor and TIL probability to the summary
    tumor = ["High tumor; low TIL;"]*(batch_size)
    return [t+summary for t in tumor]


# Get the inference on validation/train samples and save it, used further to calculate the FID.

def save_inference_results(data_loader, model, sampler, inference_config, plot=False, save=True, with_org_names=True):
    split = inference_config['split']
    number_of_steps=inference_config['steps']
    shape=inference_config['shape']
    unconditional_guidance_scale=inference_config['unconditional_guidance_scale']
    total_samples=inference_config['total_samples']
    output_dir=inference_config['output_dir']
    batch_size = data_loader.batch_size

    validation_step_outputs = []
    if total_samples == -1:
        total_samples = len(data_loader)*batch_size
    else:
        total_samples = min(total_samples, len(data_loader)*batch_size)  # Ensure we don't go over the number of samples

    output_dir = os.path.join(output_dir, f'{split}_inference')
    os.makedirs(output_dir, exist_ok=True)
    
    if plot:
        os.makedirs(os.path.join(output_dir, "original_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "sampled_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "condition_text"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "original_images2"), exist_ok=True)
        types_file = os.path.join(output_dir, "samples_types.csv")
        cell_counts_file = os.path.join(output_dir, "samples_cell_counts.csv")
        if hasattr(model, 'control_key'):
            os.makedirs(os.path.join(output_dir, model.control_key), exist_ok=True)

        
    with tqdm(total=math.ceil(total_samples/batch_size), desc=f"Running inference on {split} samples") as pbar:
        for i, samples in enumerate(data_loader):
            if i*batch_size >= total_samples:
                break
            with torch.no_grad():

                #unconditional token for classifier free guidance
                ut = get_unconditional_token(batch_size)
                uc = model.get_learned_conditioning(ut)
                
                ct = samples[model.cond_stage_key]
                cc = model.get_learned_conditioning(ct)
                if hasattr(model, 'control_key'):
                    control = samples[model.control_key]
                    if batch_size is not None:
                        control = control[:batch_size]
                    control = einops.rearrange(control, 'b h w c -> b c h w')
                    control = control.to(memory_format=torch.contiguous_format).float().to(inference_config['device'])
                    cc = dict(c_crossattn=[cc], c_concat=[control])
                    uc = dict(c_crossattn=[uc], c_concat=[control])#torch.from_numpy(np.full(control.shape, NULL_MASK, dtype=np.uint8)).to(inference_config['device'])]) # should we also pass uncoditional control?-> Tried it, but quality of images degrades, this in line with SDM and controlnet.
                samples_ddim, _ = sampler.sample(number_of_steps, batch_size, shape, cc, verbose=False, \
                                                unconditional_conditioning=uc, unconditional_guidance_scale=unconditional_guidance_scale, eta=1)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()
                validation_step_outputs.append(x_samples_ddim)
                if plot:
                    for j, image in enumerate(x_samples_ddim):
                        if not with_org_names: # retain original names from the original data source
                            suffix_1 = suffix_2 = f'{i*batch_size+j}.png'
                        else:
                                suffix_1 = samples['fname'][j]
                                suffix_2 = samples['fname2'][j]
                        to_pil_image(image.cpu()).save(os.path.join(output_dir, "sampled_images", f'T2I_{suffix_1}_M2I_{suffix_2}.png'))
                            
                        original_image = samples[model.first_stage_key][j].permute(2, 0, 1).cpu().numpy()  # Convert to numpy array
                        original_image = ((original_image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # Revert normalization and clip values
                        Image.fromarray(original_image.transpose(1, 2, 0)).save(os.path.join(output_dir, "original_images", f'T2I_{suffix_1}.png'))  # Save the reverted image
                        if 'jpg2' in samples:
                            original_image2 = samples['jpg2'][j].permute(2, 0, 1).cpu().numpy()
                            original_image2 = ((original_image2 + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                            Image.fromarray(original_image2.transpose(1, 2, 0)).save(os.path.join(output_dir, "original_images2", f'M2I_{suffix_2}.png'))  # Save the reverted image2 from dataset2
                        if hasattr(model, 'control_key'): # Assumed its image-like
                            control_j = control[j]
                            if control_j.shape[0] > 3:
                                control_j = add_edge_type_mask(control_j.unsqueeze(0)).squeeze(0)
                            Image.fromarray(control_j.permute(1,2,0).cpu().numpy().astype('uint8')).save(os.path.join(output_dir, model.control_key, f'{model.control_key}_{suffix_2}.png'))
                        with open(os.path.join(output_dir, "condition_text", f'{suffix_1}.txt'), 'w') as f:
                            f.write(samples[model.cond_stage_key][j])
            pbar.update(1)
    # Assuming validation_step_outputs is a list of tensors
    if save:
        output_file = os.path.join(output_dir, f'{split}_step_outputs.pt')
        torch.save(validation_step_outputs, output_file)
        return output_file

def compute_statistics_of_path(path):
    with np.load(path) as f:
        m, s = f["mu"][:], f["sigma"][:]
    return m, s

def get_activations(m, input_tensor, model='incpetion'):
    m.eval()
    if model == 'conch':
        with torch.inference_mode():    
            pred = m.encode_image(input_tensor, proj_contrast=False, normalize=False).cpu().numpy()
    else:
        with torch.no_grad():
            pred = m(input_tensor)
            pred = pred[0] if not isinstance(pred, torch.Tensor) else pred
            pred = pred.squeeze().cpu().numpy()
    return pred

def get_fid_model(model='inception', device='cuda:0'):
    dims={'inception': 2048, 'gigapath': 1536, 'conch': 512}
    if model == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        m = InceptionV3([block_idx])
        train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),
        ]
    )
    elif model == 'gigapath':
        m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        )
    elif model == 'conch':
        model_cfg = 'conch_ViT-B-16'
        checkpoint_path = 'conch/pytorch_model.bin'
        m, preprocessor = create_model_from_pretrained(model_cfg, checkpoint_path, force_image_size=256, device=device)
        _ = m.eval()
        m.to(device)
        return m, preprocessor, dims[model]
    else:
        raise ValueError("Invalid model")
    m.eval()
    m.to(device)
    return m, train_transforms, dims[model]

def get_embedding(image, text, model, preprocessor,  tokenizer=None, txt_sim_model='plip', device='cuda:3'):
    if txt_sim_model == 'plip':
        inputs = preprocessor(text=[text],
                    images=image,
                    return_tensors="pt",
                    padding=True, max_length=77, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
    elif txt_sim_model == 'conch':
        image = preprocessor(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            img_emb = model.encode_image(image)
            text_tokens = tokenize(texts=[text], tokenizer=tokenizer).to(device)
            text_emb = model.encode_text(text_tokens)
    return img_emb, text_emb

def get_sim_model(txt_sim_model, device):
    tokenizer=None    
    if txt_sim_model == 'plip':   
        model = CLIPModel.from_pretrained("vinid/plip")
        preprocessor = CLIPProcessor.from_pretrained("vinid/plip")
    elif txt_sim_model == 'conch':
        model_cfg = 'conch_ViT-B-16'
        checkpoint_path = 'conch/pytorch_model.bin'
        model, preprocessor = create_model_from_pretrained(model_cfg, checkpoint_path, force_image_size=256, device=device)
        tokenizer = get_tokenizer()
        _ = model.eval()
    else:
        raise ValueError("Invalid model")
    return model, preprocessor, tokenizer

def calculate_activation_statistics(images, device, batch_size=32, model='inception'):
    m, train_transforms, dims = get_fid_model(model, device)
    pred_arr = np.empty((len(images), dims))
    start_idx = 0

    for i in tqdm(range(0, len(images), batch_size), desc="Calculating FID.."):
        chunk = images[i : i + batch_size]
        if model == 'conch':
            input_tensor = torch.stack([train_transforms(Image.fromarray(img)) for img in chunk]).to(device)
        else:
            input_tensor = torch.stack([train_transforms(img) for img in chunk]).to(device)
        pred = get_activations(m, input_tensor, model)
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

from omegaconf import OmegaConf
def calculate_metrics(data, model, sampler, inference_config="plip_imagenet_finetune_inference.yaml", 
                      validation_step_outputs_file=None, fid=True, inception_score=True, fid_model='inception', cos_sim=True, txt_sim_model='plip'):
    """Calculate the FID compared against real stats given by fid_file, for the split on the dataset given on the saved model
    """
    if validation_step_outputs_file is None:
        validation_step_outputs_file = save_inference_results(data, model, sampler, inference_config=inference_config, save=True, plot=False)

    validation_step_outputs = torch.load(validation_step_outputs_file)
    all_samples = np.vstack(validation_step_outputs)
    device = inference_config['device']
    metrics = {'fid':-1, 'inception_score': -1, 'img_txt_sim_score': -1}
    if inception_score:
        inception = InceptionScore()
        inception.update(torch.from_numpy(all_samples).to(torch.uint8))
        metrics['inception_score'] = inception.compute()[0]
        print("Inception Score ", metrics['inception_score'])
    all_samples = all_samples.transpose(0, 2, 3, 1)
    if fid:
        m1, s1 = compute_statistics_of_path(inference_config['fid_path'])
        m2, s2 = calculate_activation_statistics(all_samples, device, batch_size=32, model=fid_model)
        metrics['fid'] = calculate_frechet_distance(m1, s1, m2, s2)
        print("FID ", metrics['fid'])
    if cos_sim:
        model, preprocessor, tokenizer = get_sim_model(txt_sim_model, device)
        img_folder = inference_config['sim_score_img']
        text_folder = inference_config['sim_score_txt']
        cosine_similarities = []
        for img in tqdm(os.listdir(img_folder)):
            if "_M2I" in img:
                text_f = os.path.join(text_folder, img.split("/")[-1].split("T2I_")[-1].split("_M2I")[0]+".txt")
            else:
                text_f = os.path.join(text_folder, img.split("/")[-1].split("T2I_")[-1].split(".")[0]+".txt")
            image = Image.open(os.path.join(img_folder, img))
            with open(text_f, 'r') as file:
                text = file.read()

            img_emb, text_emb = get_embedding(image, text, model, preprocessor, tokenizer, txt_sim_model, device) 
            cosine_similarity = torch.nn.functional.cosine_similarity(img_emb, text_emb)
            cosine_similarities.append(cosine_similarity.item()) 
            
        # Calculate average cosine similarity
        average_similarity = sum(cosine_similarities) / len(cosine_similarities)

        # Print the average similarity score
        print("Average Cosine Similarity:", average_similarity)
        metrics['img_txt_sim_score'] = average_similarity
    return metrics

def get_args():
    parser = argparse.ArgumentParser(description="PathDiff inference")
    parser.add_argument("--config", type=str, default="plip_imagenet_finetune_inference.yaml", help="Path to the inference config file")   
    parser.add_argument("--fid", action="store_true", help="Calculate FID")
    parser.add_argument("--fid_model", type=str, default='inception', help="FID model to use to extract features")
    parser.add_argument("--is_score", action="store_true", help="Calculate Inception score")
    parser.add_argument("--cos_sim", action="store_true", help="Calculate Image-Text similarity score")
    parser.add_argument("--txt_sim_model", type=str, default='plip', help="Text similarity model to use")
    parser.add_argument("--plot", action="store_true", help="Plot inference")
    parser.add_argument("--save", action="store_true", help="Save the inference plots") 
    parser.add_argument("--inference", action="store_true", help="Get the inference")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    config = OmegaConf.load(args.config)
    data_config_params = OmegaConf.load(config['data_config'])
    if config.split == 'train':
        data = instantiate_from_config(data_config_params.data.params.train)
    else:
        data = instantiate_from_config(data_config_params.data.params.validation)
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
      
    data = DataLoader(data, batch_size = data_config_params.data.params.batch_size,
            num_workers=data_config_params.data.params.num_workers,
            shuffle=False,
            drop_last=True
        )
    model = get_model(config['model_config'], config['device'], config['ckpt_path'])
    sampler = DDIMSampler(model)
    os.makedirs(config['output_dir'], exist_ok=True)
    OmegaConf.save(config, os.path.join(config['output_dir'], "inference_config.yaml"))
    OmegaConf.save(data_config_params, os.path.join(config['output_dir'], "data_config.yaml"))
    OmegaConf.save(OmegaConf.load(config['model_config']), os.path.join(config['output_dir'], "model_config.yaml"))
    
    if args.inference:
        validation_step_outputs_file = save_inference_results(data, model, sampler, inference_config=config, plot=args.plot, save=args.save)       
    else:
        validation_step_outputs_file = config['validation_step_outputs_file']