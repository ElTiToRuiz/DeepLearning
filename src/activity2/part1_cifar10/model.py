import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet adapted to CIFAR-10 (32x32, 10 classes).

    Differences from the original AlexNet (ImageNet, 224x224):
    - 3x3 kernels instead of 11x11 in the first layer (images are much smaller)
    - 3 pooling layers instead of 5 (you can't reduce that much from 32x32)
    - Same idea: 5 convolutional blocks + 3 fully connected layers
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Feature extractor: 5 conv blocks
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 4: 8x8
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 5: 8x8 -> 4x4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier: 3 FC layers with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
