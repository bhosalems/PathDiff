# PathDiff

**Histopathology Image Synthesis with Unpaired Text and Mask Conditions**

---

## 📖 Overview

PathDiff is a novel diffusion-based framework for generating high-quality histopathology images by jointly leveraging **unpaired** text reports and cell-type masks. Unlike prior methods that require paired annotations, PathDiff learns from two separate datasets—one with image–text pairs and one with image–mask pairs—and at inference time can synthesize images conditioned on:

- **Text** only  
- **Mask** only  
- **Both text & mask**  

<p align="center">
  <img src="figures/PathDiff_Method.png" alt="PathDiff Method" width="80%"/>
</p>

*Figure 1. PathDiff training & inference pipeline.*

---

## ⚙️ Installation

```bash
git clone https://github.com/bhosalems/PathDiff.git
cd PathDiff
# create and activate a conda/env virtual environment
conda env create -f environment.yml
conda activate pathdiff

### Quick Start

```bash
# Download Pretrained Models
wget https://github.com/bhosalems/PathDiff/releases/download/v1.0/pathdiff_mask2img.pth -P models/
wget https://github.com/bhosalems/PathDiff/releases/download/v1.0/pathdiff_text2img.pth -P models/

# Run Inference
python scripts/inference.py \
  --model models/pathdiff_mask2img.pth \
  --mask path/to/cell_mask.png \
  --text "High-grade carcinoma with prominent nucleoli" \
  --output results/output.png
```

#### Arguments

```text
--model   : Path to the checkpoint file (e.g., models/pathdiff_mask2img.pth)
--mask    : 256×256 binary or semantic mask (e.g., path/to/cell_mask.png)
--text    : Diagnostic report or caption (e.g., "High-grade carcinoma with prominent nucleoli")
--output  : Output image path (e.g., results/output.png)
```

### 🖼 Qualitative Results

```markdown
<p align="center">
  <img src="figures/qualitative_results.png" alt="Qualitative Results" width="90%"/>
</p>
*Figure 2. Sample syntheses on PanNuke, CoNIC, and MoNuSAC.*
```

### 📊 Quantitative Results

```markdown
| Dataset    | Condition    | FID ↓    | KID ↓    | PLIP ↑   |
|------------|--------------|---------:|---------:|---------:|
| PanNuke    | Mask only    | **7.21** | 0.0415   | –        |
| PathCap    | Text only    | **14.04**| 0.0557   | **24.66**|
| PathCap    | Mask+Text    | **10.54**| 0.0766   | **24.02**|

Table 1. Main generation metrics across different conditioning modes.
```

### 🗄 Model Zoo

```markdown
| Task            | Checkpoint                                          | Notes                               |
|-----------------|-----------------------------------------------------|-------------------------------------|
| Mask → Image    | pathdiff_mask2img.pth (PanNuke+CoNIC+MoNuSAC)       | Trained on three M2I datasets       |
| Text → Image    | pathdiff_text2img.pth (PathCap subset)              | Trained on PathCap histogram images |
| Unified Model   | pathdiff_unified.pth (Joint T2I & M2I)              | Joint training on both modalities   |
```

### 🔧 Scripts

```markdown
- scripts/train.py       # Training entrypoint (coming soon)
- scripts/inference.py   # Run generation from text, mask, or both
- scripts/evaluate.py    # Compute FID, KID, PLIP, Faithfulness
```

### 🤝 Contributing

```markdown
1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/foo`)  
3. Commit your changes (`git commit -m 'Add foo'`)  
4. Push to the branch (`git push origin feature/foo`)  
5. Open a Pull Request  

Please see CONTRIBUTING.md for more details.
```

### 📑 Citation

```bibtex
@inproceedings{bhosale2025pathdiff,
  title     = {PathDiff: Histopathology Image Synthesis with Unpaired Text and Mask Conditions},
  author    = {Bhosale, Mahesh and Wasi, Abdul and Zhai, Yuanhao and Tian, Yunjie and Border, Samuel and Xi, Nan and Sarder, Pinaki and Yuan, Junsong and Doermann, David and Gong, Xuan},
  booktitle = {Proceedings of ICCV},
  year      = {2025}
}
```

### 📄 License

```markdown
This project is licensed under the MIT License – see the LICENSE file for details.
```
