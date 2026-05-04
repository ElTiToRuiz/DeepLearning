"""
PARTE 2 — Modelo 2: VGG16 con Transfer Learning en Chest X-Ray
===============================================================
En vez de aprender desde cero, aprovechamos que VGG16 ya fue
entrenada con 1.2 millones de imágenes de ImageNet.

La clave del transfer learning en PyTorch:
  1. Cargamos VGG16 con pesos pre-entrenados
  2. "Congelamos" todas sus capas → sus pesos NO cambian
  3. Sustituimos solo la última capa por una nueva (2 clases)
  4. Entrenamos únicamente esa última capa nueva

Esto es mucho más rápido y generalmente da mejores resultados
cuando el dataset no es enorme.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time

from part2_medical.data.dataset import get_dataloaders

# ─── Configuración ────────────────────────────────────────────────────────────

# ⚠️ CAMBIA ESTA RUTA por la tuya
DATA_DIR    = "C:\Users\PC\Downloads\chest_xray"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
NUM_EPOCHS  = 10    # Con transfer learning necesitamos MUCHAS menos épocas
LR          = 0.0001  # Learning rate más pequeño: los pesos ya están bien ajustados
NUM_CLASSES = 2

print(f"Usando dispositivo: {DEVICE}")

# ─── Cargar VGG16 pre-entrenada y adaptarla ───────────────────────────────────

def build_vgg16_transfer(num_classes=2):
    """
    Construye el modelo VGG16 adaptado para transfer learning.
    
    VGG16 original tiene esta estructura:
      features   → 13 capas convolucionales (el extractor de características)
      avgpool    → adaptive average pooling
      classifier → 3 capas FC que terminan en 1000 clases (ImageNet)
    
    Nosotros:
      1. Cargamos todo con pesos de ImageNet
      2. Congelamos 'features' y 'avgpool' → no se entrenan
      3. Reemplazamos el 'classifier' final por uno de 2 clases
    """

    # weights='DEFAULT' descarga los pesos pre-entrenados en ImageNet
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # ── Paso clave: CONGELAR las capas convolucionales ──────────────────────
    # requires_grad=False significa que PyTorch NO calculará gradientes
    # para estos parámetros → no se actualizan durante el entrenamiento
    for param in vgg.features.parameters():
        param.requires_grad = False

    # ── Sustituir el clasificador final ─────────────────────────────────────
    # vgg.classifier es un Sequential con 6 capas que termina en Linear(4096, 1000)
    # Lo reemplazamos por uno que termine en 2 clases (NORMAL / PNEUMONIA)
    in_features = vgg.classifier[6].in_features   # = 4096

    vgg.classifier[6] = nn.Linear(in_features, num_classes)
    # ↑ Esta es la ÚNICA capa nueva. Solo ella se entrenará desde cero.

    return vgg


model = build_vgg16_transfer(NUM_CLASSES).to(DEVICE)

# Solo mostramos los parámetros que SÍ se entrenan
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nParámetros totales:      {total:,}")
print(f"Parámetros entrenables:  {trainable:,}  ({100*trainable/total:.1f}%)")
print("→ El resto están congelados (pesos de ImageNet)")

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


# ─── Entrenamiento ────────────────────────────────────────────────────────────

train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

# Solo optimizamos los parámetros entrenables (la capa final)
# filter(lambda p: p.requires_grad, ...) → ignora los congelados
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n─── Entrenando VGG16 con Transfer Learning en Chest X-Ray ───")
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

# ─── Evaluación final ─────────────────────────────────────────────────────────

_, test_acc, preds, labels = evaluate(model, test_loader, criterion)
print(f"\nPrecisión en TEST: {test_acc:.2f}%")
print("\nReporte completo:")
print(classification_report(labels, preds, target_names=classes))

# ─── Guardar modelo y gráficas ────────────────────────────────────────────────

os.makedirs('./part2_medical/results', exist_ok=True)
torch.save(model.state_dict(), './part2_medical/results/vgg16_transfer.pth')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue')
ax1.plot(epochs, history['val_loss'],   label='Validación', color='tomato')
ax1.set_title('Pérdida — VGG16 Transfer Learning')
ax1.set_xlabel('Época'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history['train_acc'], label='Train', color='steelblue')
ax2.plot(epochs, history['val_acc'],   label='Validación', color='tomato')
ax2.set_title('Precisión — VGG16 Transfer Learning')
ax2.set_xlabel('Época'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('VGG16 Transfer Learning — Chest X-Ray (Neumonía)', fontsize=13)
plt.tight_layout()
plt.savefig('./part2_medical/results/vgg16_transfer_training.png', dpi=150)
plt.show()