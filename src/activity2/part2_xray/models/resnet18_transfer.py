import torch.nn as nn
import torchvision.models as models


def build_resnet18_transfer(num_classes: int = 2) -> nn.Module:
    """
    ResNet18 pretrained on ImageNet, adapted via transfer learning:

      1. Load pretrained weights (1.2M images, 1000 classes).
      2. Freeze every convolutional layer (`requires_grad=False`)
         -> these weights are not updated during training.
      3. Replace the final `fc` layer (Linear(512, 1000)) with a new
         Linear(512, num_classes). That is the only layer being trained.

    ResNet18 (~11M params) is much lighter than VGG16 (~138M params) and
    trains substantially faster on hardware without CUDA (e.g. Apple Silicon
    with MPS), while matching or beating VGG16's accuracy.
    """
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything pretrained.
    for param in resnet.parameters():
        param.requires_grad = False

    # The head is a single Linear: replace it.
    in_features = resnet.fc.in_features  # = 512
    resnet.fc = nn.Linear(in_features, num_classes)
    # Only `fc` keeps requires_grad=True by default.

    return resnet


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable parameters, total parameters)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total
