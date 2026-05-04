import os
import torch

# Paths anchored to this file so the activity works from any cwd.
ACTIVITY_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR     = os.path.join(ACTIVITY_DIR, "dataset")
MODELS_DIR      = os.path.join(ACTIVITY_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(ACTIVITY_DIR, "results")

# --- Part 1: CIFAR-10 ---
CIFAR10_DIR     = os.path.join(DATASET_DIR, "cifar10")  # torchvision downloads here
CIFAR10_NUM_CLASSES = 10
CIFAR10_BATCH_SIZE  = 128
CIFAR10_EPOCHS      = 50
CIFAR10_MAX_LR      = 5e-3   # OneCycleLR peak
CIFAR10_WEIGHT_DECAY = 1e-4

# --- Part 2: Chest X-Ray ---
# The dataset is downloaded via kagglehub into ~/.cache/kagglehub/.
# See src/activity2/part2_xray/dataset.py:download_xray_dataset()
XRAY_NUM_CLASSES = 2
XRAY_IMG_SIZE   = 224
XRAY_BATCH_SIZE = 32
# Scratch
XRAY_SCRATCH_EPOCHS       = 15
XRAY_SCRATCH_MAX_LR       = 3e-3   # OneCycleLR peak
XRAY_SCRATCH_WEIGHT_DECAY = 1e-4
# Transfer learning
XRAY_TRANSFER_EPOCHS = 10
XRAY_TRANSFER_LR     = 1e-4

# --- Reproducibility ---
RANDOM_SEED = 42


def get_device() -> torch.device:
    """
    Pick the best available backend:
      1. CUDA (NVIDIA)
      2. MPS  (Apple Silicon — Metal)
      3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
