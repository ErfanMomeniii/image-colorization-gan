import os

# Paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
TRAINED_MODELS_DIR = "trained_models"  # Final trained models (in .gitignore)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.5
BETA2 = 0.999

# Model parameters
IMAGE_SIZE = 256
L1_LAMBDA = 100  # Weight for L1 loss in generator

# Device - auto-detect best available
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Logging
SAVE_EVERY = 5  # Save checkpoint every N epochs
SAMPLE_EVERY = 1  # Generate sample images every N epochs
