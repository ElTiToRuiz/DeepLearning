import random
import numpy as np
import torch
 
# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
DATA_PATH   = "data/insurance.csv"
MODELS_DIR  = "models/"
RESULTS_DIR = "results/"
 
# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
RANDOM_SEED = 42
 
# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
TEST_SIZE = 0.2
 
# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────
EPOCHS        = 200
LEARNING_RATE = 0.01
BATCH_SIZE    = 32
 
# ─────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────
SHALLOW_HIDDEN_SIZE = 32
DEEP_HIDDEN_SIZES   = [64, 32, 16]
DROPOUT_RATE        = 0.2
 
# ─────────────────────────────────────────
# OPTUNA
# ─────────────────────────────────────────
OPTUNA_TRIALS = 20
 
 
def set_seeds(seed: int = RANDOM_SEED):
    """
    Sets all random seeds to ensure reproducibility.
    Always call at the beginning of pipeline.py before any other operation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[CONFIG] Seeds set to {seed}")