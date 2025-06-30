import warnings
import numpy as np
import torch
from torchvision import transforms
warnings.filterwarnings("ignore")
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
import timm
import os
from pytorch_fid.fid_score import calculate_frechet_distance
from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from cleanfid import fid as clean_fid
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

def get_fid_model(model='inception', device='cuda:4'):
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

def compute_statistics_of_path(path):
    with np.load(path) as f:
        m, s = f["mu"][:], f["sigma"][:]
    return m, s

def get_embedding(image, text, model, preprocessor,  tokenizer=None, txt_sim_model='plip', device='cuda:4'):
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

def calculate_metrics(syn_image_files_dir, real_image_files_dir, text_dir=None, fid=True, inception_score=True, fid_model='inception', device='cuda:4', cos_sim=False, txt_sim_model='plip', num_files = 1024):
    """Calculate the FID compared against real stats given by fid_file, for the split on the dataset given on the saved model
    """
    metrics = {'fid':-1, 'inception_score': -1}
    # print("Calculating metric on " + str(len(syn_images)) + " images..")
    # if inception_score:
    #     inception = InceptionScore()
    #     inception.update(torch.from_numpy(syn_images).to(torch.uint8))
    #     metrics['inception_score'] = inception.compute()[0]
    #     print("Inception Score ", metrics['inception_score'])
    # syn_images = syn_images.transpose(0, 2, 3, 1)
    if fid:
        metrics['kid'] = clean_fid.compute_kid(syn_image_files_dir, real_image_files_dir, mode='clean', device=torch.device(device), use_dataparallel=False)
        metrics['clean_fid'] = clean_fid.compute_fid(syn_image_files_dir, real_image_files_dir,  mode="clean", device=torch.device(device), use_dataparallel=False)
        metrics['clip_fid'] = clean_fid.compute_fid(syn_image_files_dir, real_image_files_dir,  mode="clean", model_name="clip_vit_b_32", device=torch.device(device), use_dataparallel=False)

    if cos_sim:
        model, preprocessor, tokenizer = get_sim_model(txt_sim_model, device)
        cosine_similarities = []
        empty_text_files = 0
        not_found = 0
        textdf = None
        if not os.path.isdir(text_dir):
            textdf = pd.read_csv(text_dir)
        
        for img in tqdm(os.listdir(syn_image_files_dir)):
            if textdf is None:
                if "_M2I" in img:
                    text_f = os.path.join(text_dir, img.split("/")[-1].split("T2I_")[-1].split("_M2I")[0]+".txt")
                elif "_T2I" in img:
                    text_f = os.path.join(text_dir, img.split("/")[-1].split("T2I_")[-1].split(".")[0]+".txt")
                else:
                    text_f = os.path.join(text_dir, img.split("/")[-1].split(".")[0].split("_")[1]+".txt")
                image = Image.open(os.path.join(syn_image_files_dir, img))
                with open(text_f, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # if (len(text) == 0): # skip empty files
                    #     empty_text_files+=1
                    #     continue
            else:
                image = Image.open(os.path.join(syn_image_files_dir, img))
                text = textdf.loc[textdf['img'].str.contains(os.path.splitext(img)[0])]['text'].values[0]
            img_emb, text_emb = get_embedding(image, text, model, preprocessor, tokenizer, txt_sim_model, device) 
            cosine_similarity = torch.nn.functional.cosine_similarity(img_emb, text_emb)
            cosine_similarities.append(cosine_similarity.item()) 
            
        # Calculate average cosine similarity
        print("skipped {} not found text files".format(not_found))
        print("skipped {} empty text files".format(empty_text_files))
        metrics['average_cos_similarity'] = sum(cosine_similarities) / len(cosine_similarities)
    return metrics


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

if __name__ == "__main__":
    metrics = calculate_metrics("/home/csgrad/mbhosale/train_inference/carcinoma_images", "/home/csgrad/mbhosale/train_inference/original_images", "/home/csgrad/mbhosale/train_inference/carcinoma_text", fid=True, inception_score=False, cos_sim=True, device='cuda', fid_model='clip')
    print(metrics)