import os
import torch
from src.config.config import MODELS_DIR
from src.logger.logger import logger
 
 
def save_model(model, model_name: str):
    """
    Saves just the weights (state_dict). 
    Remember: you need to build the model structure first before loading these back.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Saved weights to: {path}")
 
 
def load_model(model, model_name: str):
    """
    Injects saved weights back into a model. 
    Make sure the architecture matches or this will crash!
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find the model at: {path}")
 
    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info(f"Loaded weights from: {path}")
    return model