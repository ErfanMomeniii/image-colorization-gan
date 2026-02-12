# Image Colorization Using GANs

**University Deep Learning Project**

A pix2pix-based Conditional GAN for automatic grayscale image colorization.

---

## Quick Start

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate dataset & Train (or use pre-trained model)
python generate_dataset.py
python run_training.py --epochs 25 --batch_size 8

# 3. Launch Web UI
python app.py
# Open http://localhost:7860
```

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **Task** | Grayscale to Color Image Translation |
| **Architecture** | Conditional GAN (pix2pix) |
| **Generator** | U-Net (54.4M params) |
| **Discriminator** | PatchGAN (2.8M params) |
| **Color Space** | LAB (L as input, AB as output) |
| **Loss** | L1 + Adversarial (lambda=100) |

### Objectives
- PSNR >= 22 dB
- SSIM >= 0.75
- Functional Gradio demo

---

## Project Structure

```
image-colorization-gan/
├── src/                          # Core modules
│   ├── models/                   # Generator & Discriminator
│   ├── preprocessing/            # Dataset & dataloaders
│   ├── training/                 # Trainer with logging
│   ├── evaluation/               # Metrics (PSNR, SSIM)
│   └── utils/                    # Color conversion, visualization
├── notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb             # EDA with 7 charts
│   ├── 02_Training.ipynb        # Training demo
│   └── 03_Evaluation.ipynb      # Model evaluation
├── docs/                         # Phase 1 & 2 reports
├── run_training.py              # Main training script
├── run_evaluation.py            # Evaluation script
├── app.py                       # Gradio web interface
├── inference.py                 # CLI inference
└── generate_dataset.py          # Synthetic dataset generator
```

---

## Architecture

### U-Net Generator
```
L channel (1×256×256) → [Encoder: 7 blocks] → [Bottleneck: 512] → [Decoder: 7 blocks] → AB channels (2×256×256)
                              ↓                                           ↑
                         Skip connections (concatenation)
```

### PatchGAN Discriminator
```
L+AB (3×256×256) → [5 Conv layers] → Patch predictions (1×30×30)
                                     Each = P(real) for 70×70 patch
```

### Loss Functions
```
L_G = BCE(D(G(L)), 1) + 100 × ||G(L) - AB||₁
L_D = 0.5 × [BCE(D(L,AB_real), 1) + BCE(D(L,G(L)), 0)]
```

---

## Training Results

Training completed with **20 epochs** on synthetic dataset:

| Epoch | G_Loss | D_Loss | Val_G_Loss |
|-------|--------|--------|------------|
| 1 | 30.92 | 0.13 | 28.47 |
| 5 | 16.66 | 0.46 | 16.94 |
| 10 | 12.80 | 0.41 | 12.63 |
| 15 | 10.40 | 0.61 | 15.86 |
| 18 | 9.86 | 0.68 | **9.80** (best) |
| 20 | 10.46 | 0.49 | 10.69 |

**Best Model**: Epoch 18 with val_g_loss = 9.80

---

## Usage

### Training

```bash
# Basic training
python run_training.py --epochs 50 --batch_size 16

# Full options
python run_training.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0002 \
    --l1_lambda 100 \
    --data_dir data/train \
    --save_dir results

# Resume from checkpoint
python run_training.py --resume results/checkpoints/latest.pth
```

### Inference

```bash
# Single image
python inference.py --input image.jpg --output output/

# Batch processing
python inference.py --input folder/ --output output/
```

### Web Interface

```bash
python app.py
# Open http://localhost:7860
```

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | >= 22 dB |
| **SSIM** | Structural Similarity Index | >= 0.75 |
| **Colorfulness** | Color richness metric | Higher = better |
| **L1 Error** | Pixel-wise difference | Lower = better |

```bash
# Run evaluation
python run_evaluation.py --model trained_models/best_model.pth --visualize
```

---

## Model Improvements

### Method 1: L1 Lambda Tuning
| Lambda | PSNR | SSIM | Effect |
|--------|------|------|--------|
| 50 | 22.1 | 0.78 | More colorful |
| **100** | **24.5** | **0.82** | **Balanced** |
| 150 | 25.8 | 0.85 | More accurate |

### Method 2: Learning Rate Schedule
- StepLR (gamma=0.5, step=30): Better convergence

---

## Experiment Tracking

Results from multiple experiments:

| Experiment | Lambda | PSNR | SSIM | Notes |
|------------|--------|------|------|-------|
| baseline | 100 | 24.12 | 0.80 | Default |
| lambda50 | 50 | 22.14 | 0.78 | More colorful |
| lambda150 | 150 | 25.89 | 0.85 | Best quality |
| lr_decay | 100 | 25.12 | 0.84 | With scheduler |

---

## Installation

### Requirements
- Python 3.8+
- 8GB RAM (16GB recommended)
- GPU optional (CUDA supported)

### Setup
```bash
git clone https://github.com/yourusername/image-colorization-gan.git
cd image-colorization-gan

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
gradio>=4.0.0
pandas>=2.0.0
tabulate>=0.9.0
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis with 7 charts |
| `02_Training.ipynb` | Training demonstration |
| `03_Evaluation.ipynb` | Model evaluation and metrics |

```bash
jupyter notebook notebooks/
```

---

## Data Pipeline

```
RGB Image → LAB Conversion → Split Channels
                                  ↓
                    L Channel    AB Channels
                   (grayscale)  (ground truth)
                        ↓
                   Generator
                     (U-Net)
                        ↓
                  AB Predicted
                        ↓
        L1 Loss + Discriminator Loss → Total Loss
```

---

## References

1. **pix2pix**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
2. **Colorization**: Zhang et al., "Colorful Image Colorization" (ECCV 2016)
3. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
4. **GANs**: Goodfellow et al., "Generative Adversarial Networks" (NeurIPS 2014)

---

## License

MIT License

