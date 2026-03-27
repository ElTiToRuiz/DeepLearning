import random
import numpy as np
import torch
 
# --- Paths to our stuff ---
DATA_PATH   = "data/insurance.csv"
MODELS_DIR  = "models/"
RESULTS_DIR = "results/"
 
# --- Keeping things consistent ---
RANDOM_SEED = 42
 
# --- Data tweaks ---
TEST_SIZE = 0.2
 
# --- Training basics ---
EPOCHS        = 200
LEARNING_RATE = 0.01
BATCH_SIZE    = 32
 
# --- Model structure defaults ---
SHALLOW_HIDDEN_SIZE = 32
DEEP_HIDDEN_SIZES   = [64, 32, 16]
DROPOUT_RATE        = 0.2
 
# --- Optuna search settings ---
OPTUNA_TRIALS = 20
 
 
def set_seeds(seed: int = RANDOM_SEED):
    """
    Lock in the seeds for everyone so we get the same results 
    every time we run the script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[CONFIG] Seeds locked to {seed}")