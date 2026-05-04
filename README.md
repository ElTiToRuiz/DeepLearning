# Deep Learning Assignment — Computer Vision

## Estructura del proyecto

```
deep_learning_assignment/
│
├── part1_alexnet/
│   └── alexnet_cifar10.py       # CNN desde cero con CIFAR-10
│
├── part2_medical/
│   ├── models/
│   │   ├── cnn_scratch.py       # CNN desde cero para rayos X
│   │   └── vgg_transfer.py      # VGG16 con transfer learning
│   ├── data/
│   │   └── dataset.py           # Carga y preprocesado del dataset
│   └── results/                 # Gráficas y métricas guardadas
│
├── utils/
│   └── trainer.py               # Funciones de entrenamiento y evaluación
│
├── requirements.txt
└── README.md
```

## Dataset

- **Parte 1:** CIFAR-10 (descarga automática con torchvision)
- **Parte 2:** Chest X-Ray Images (Pneumonia) — Kaggle
  - Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - Descargar y colocar en `part2_medical/data/chest_xray/`

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
# Parte 1 — AlexNet en CIFAR-10
python part1_alexnet/alexnet_cifar10.py

# Parte 2 — CNN desde cero en rayos X
python part2_medical/models/cnn_scratch.py

# Parte 2 — VGG16 con transfer learning en rayos X
python part2_medical/models/vgg_transfer.py
```