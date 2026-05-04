import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
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
