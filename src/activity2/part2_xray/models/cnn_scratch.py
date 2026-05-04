import torch.nn as nn


class CNNScratch(nn.Module):
    """
    CNN trained from scratch for Chest X-Ray classification (224x224x3 -> 2 classes).

    Efficient design:
      - 4 conv blocks (32, 64, 128, 256) with BatchNorm + MaxPool
      - **Global Average Pooling** instead of a giant Linear (50176->512) which
        would have ~25M parameters. With GAP the head is ~131K params, ~200x
        fewer, trains faster, and tends to generalize better (less prone to
        overfitting than a huge FC layer).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112 — edges and textures
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56 — more complex patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28 — structures (ribs, tissue)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 -> 14 — high-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling: 256x14x14 -> 256x1x1, then flatten -> 256
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Compact classifier: 256 -> 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
