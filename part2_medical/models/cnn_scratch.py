"""
PARTE 2 — Modelo 1: CNN desde cero en Chest X-Ray
==================================================
Usamos una arquitectura similar a AlexNet pero entrenada
desde cero sobre el dataset de rayos X.

Esto nos servirá como línea base (baseline) para comparar
con el transfer learning de VGG16.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

from part2_medical.data.dataset import get_dataloaders

# ─── Configuración ────────────────────────────────────────────────────────────

# ⚠️ CAMBIA ESTA RUTA por la tuya
DATA_DIR    = "C:\Users\PC\Downloads\chest_xray"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 0.001
NUM_CLASSES = 2   # NORMAL vs PNEUMONIA

print(f"Usando dispositivo: {DEVICE}")

# ─── Arquitectura CNN desde cero ──────────────────────────────────────────────

class CNNScratch(nn.Module):
    """
    CNN desde cero para clasificación de rayos X.
    
    Entrada: imágenes 224x224x3
    Salida:  2 clases (NORMAL / PNEUMONIA)
    
    Arquitectura inspirada en AlexNet pero simplificada:
    - 4 bloques convolucionales en vez de 5
    - Menos filtros para adaptarse al tamaño del dataset
    """
    def __init__(self, num_classes=2):
        super(CNNScratch, self).__init__()

        self.features = nn.Sequential(
            # Bloque 1: detecta bordes y texturas simples
            # 224x224 → 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),    # Normaliza activaciones → entrenamiento más estable
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloque 2: detecta patrones más complejos
            # 112x112 → 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloque 3: detecta estructuras (costillas, tejidos...)
            # 56x56 → 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloque 4: características de alto nivel
            # 28x28 → 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Clasificador final
        # Entrada: 256 canales × 14×14 píxeles = 50176 valores
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Aplanar
        x = self.classifier(x)
        return x


# ─── Funciones de entrenamiento ───────────────────────────────────────────────

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
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels


# ─── Entrenamiento principal ──────────────────────────────────────────────────

train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

model     = CNNScratch(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

print(f"\nParámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n─── Entrenando CNN desde cero en Chest X-Ray ───")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

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

# ─── Evaluación final en test ─────────────────────────────────────────────────

_, test_acc, preds, labels = evaluate(model, test_loader, criterion)
print(f"\nPrecisión en TEST: {test_acc:.2f}%")
print("\nReporte completo:")
print(classification_report(labels, preds, target_names=classes))

# ─── Guardar modelo y gráficas ────────────────────────────────────────────────

os.makedirs('./part2_medical/results', exist_ok=True)
torch.save(model.state_dict(), './part2_medical/results/cnn_scratch.pth')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue')
ax1.plot(epochs, history['val_loss'],   label='Validación', color='tomato')
ax1.set_title('Pérdida — CNN desde cero')
ax1.set_xlabel('Época'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history['train_acc'], label='Train', color='steelblue')
ax2.plot(epochs, history['val_acc'],   label='Validación', color='tomato')
ax2.set_title('Precisión — CNN desde cero')
ax2.set_xlabel('Época'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('CNN desde cero — Chest X-Ray (Neumonía)', fontsize=13)
plt.tight_layout()
plt.savefig('./part2_medical/results/cnn_scratch_training.png', dpi=150)
plt.show()