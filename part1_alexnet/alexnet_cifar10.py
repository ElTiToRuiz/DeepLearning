"""
PARTE 1 — CNN desde cero: AlexNet en CIFAR-10
==============================================
AlexNet fue originalmente diseñada para ImageNet (224x224).
Aquí la adaptamos a CIFAR-10 (32x32, 10 clases).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ─── Configuración ────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 128
NUM_EPOCHS  = 30
LR          = 0.001
NUM_CLASSES = 10

print(f"Usando dispositivo: {DEVICE}")

# ─── Dataset CIFAR-10 ─────────────────────────────────────────────────────────

# Transformaciones: normalización con media/std estándar de CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Data augmentation
    transforms.RandomCrop(32, padding=4),       # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASSES = ('avión', 'coche', 'pájaro', 'gato', 'ciervo',
           'perro', 'rana', 'caballo', 'barco', 'camión')

# ─── Arquitectura AlexNet adaptada a CIFAR-10 ─────────────────────────────────

class AlexNet(nn.Module):
    """
    AlexNet adaptada para imágenes 32x32 (CIFAR-10).
    
    Diferencias respecto al AlexNet original (ImageNet):
    - Kernels más pequeños en la primera capa (3x3 en vez de 11x11)
    - Menos capas de pooling (las imágenes son mucho más pequeñas)
    - Mismo concepto: bloques Conv → ReLU → Pool → capas FC al final
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # Bloque de convoluciones (extractor de características)
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32 → 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 32x32 → 16x16

            # Bloque 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),          # 16x16 → 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16x16 → 8x8

            # Bloque 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # 8x8 → 8x8
            nn.ReLU(inplace=True),

            # Bloque 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # 8x8 → 8x8
            nn.ReLU(inplace=True),

            # Bloque 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # 8x8 → 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 8x8 → 4x4
        )

        # Capas totalmente conectadas (clasificador)
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
        x = x.view(x.size(0), -1)  # Aplanar: [batch, 256*4*4]
        x = self.classifier(x)
        return x


model = AlexNet(num_classes=NUM_CLASSES).to(DEVICE)
print(f"\nParámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ─── Optimizador y función de pérdida ─────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Reduce el LR si la pérdida no mejora (ayuda a converger mejor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


# Historial para las gráficas
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   []
}

print("\n─── Entrenando AlexNet en CIFAR-10 ───")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc     = evaluate(model, test_loader, criterion)

    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Época {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

total_time = time.time() - start_time
print(f"\nEntrenamiento completado en {total_time/60:.1f} minutos")
print(f"Precisión final en test: {history['val_acc'][-1]:.2f}%")
print(f"(Referencia AlexNet en CIFAR-10: ~85-90%)")

# ─── Guardar modelo ───────────────────────────────────────────────────────────

os.makedirs('./results', exist_ok=True)
torch.save(model.state_dict(), './results/alexnet_cifar10.pth')
print("Modelo guardado en ./results/alexnet_cifar10.pth")

# ─── Gráficas ─────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)

# Pérdida
ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue')
ax1.plot(epochs, history['val_loss'],   label='Validación', color='tomato')
ax1.set_title('Pérdida (Loss)')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Precisión
ax2.plot(epochs, history['train_acc'], label='Train', color='steelblue')
ax2.plot(epochs, history['val_acc'],   label='Validación', color='tomato')
ax2.set_title('Precisión (Accuracy)')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('AlexNet en CIFAR-10 — Entrenamiento desde cero', fontsize=13)
plt.tight_layout()
plt.savefig('./results/alexnet_cifar10_training.png', dpi=150)
plt.show()
print("Gráfica guardada en ./results/alexnet_cifar10_training.png")