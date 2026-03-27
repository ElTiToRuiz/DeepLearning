import os
import torch
from src.config.config import MODELS_DIR
from src.logger.logger import logger
 
 
def save_model(model, model_name: str):
    """
    Saves only the model weights (state_dict).
    To load, you need to instantiate the architecture first.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved at: {path}")
 
 
def load_model(model, model_name: str):
    """
    Loads weights into a model with the same architecture.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
 
    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info(f"Model loaded from: {path}")
    return model