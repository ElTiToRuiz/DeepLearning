import os
import torch

from src.shared.logger import logger


def save_model(model, model_name: str, models_dir: str):
    """
    Saves just the weights (state_dict).
    Remember: you need to build the model structure first before loading these back.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Saved weights to: {path}")


def load_model(model, model_name: str, models_dir: str):
    """
    Injects saved weights back into a model.
    Make sure the architecture matches or this will crash!
    """
    path = os.path.join(models_dir, f"{model_name}.pth")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find the model at: {path}")

    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info(f"Loaded weights from: {path}")
    return model
