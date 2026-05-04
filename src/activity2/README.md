# Activity 2 — Computer Vision

Two parts:

1. **AlexNet from scratch on CIFAR-10** (Part 1, validate the architecture learns)
2. **CNN from scratch vs ResNet18 transfer learning on Chest X-Ray** (Part 2, comparison)

## Layout

```
src/activity2/
├── config.py                    # paths, hyperparameters, MPS/CUDA detection
├── train.py                     # train_epoch + evaluate + fit (shared)
├── plots.py                     # curves, confusion matrix, comparison
├── part1_cifar10/
│   ├── model.py                 # AlexNet adapted to 32x32
│   ├── pipeline.py
│   └── main.py
├── part2_xray/
│   ├── dataset.py               # ImageFolder + transforms + 10% val split
│   ├── models/
│   │   ├── cnn_scratch.py       # 4-block CNN with BatchNorm + GAP
│   │   └── resnet18_transfer.py # ResNet18 with frozen features
│   ├── pipeline_scratch.py
│   ├── pipeline_transfer.py
│   └── main.py
├── dataset/                     # gitignored
├── checkpoints/                 # gitignored
└── results/                     # gitignored
```

## How to run

### Part 1 — AlexNet on CIFAR-10
```bash
python -m src.activity2.part1_cifar10.main
```
CIFAR-10 is downloaded automatically the first time (Toronto server with a
kagglehub fallback if it is down). ~50 epochs with OneCycleLR, ~30-40 min on
M2 with MPS. Reference accuracy: ~85-90%.

### Part 2 — Chest X-Ray
The dataset is downloaded automatically via `kagglehub` the first time
(~2.3 GB, cached under `~/.cache/kagglehub/`). No manual action required.

```bash
# Train both models and print the comparison
python -m src.activity2.part2_xray.main

# Or run them separately:
python -m src.activity2.part2_xray.pipeline_scratch
python -m src.activity2.part2_xray.pipeline_transfer
```

> Note: the first run will ask for Kaggle credentials if they are not already
> configured. `kagglehub` walks you through it (browser + API token).

## Re-running without re-training

Once trained, every model is saved under `src/activity2/checkpoints/`:

- `alexnet_cifar10.pth`
- `cnn_scratch_xray.pth`
- `resnet18_transfer_xray.pth`

To re-run a pipeline (regenerate plots, recompute test metrics) **without
retraining**, set `SKIP_TRAINING=1`:

```bash
SKIP_TRAINING=1 python -m src.activity2.part1_cifar10.main
SKIP_TRAINING=1 python -m src.activity2.part2_xray.main
```

The pipeline loads the saved weights, skips the training loop, and runs the
final evaluation + confusion matrix. Takes seconds instead of hours.

If `SKIP_TRAINING=1` is set but the checkpoint is missing, the pipeline
falls back to a full training run (with a warning).

## Hardware

`config.py` automatically picks the best available device:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU

On an M2 with MPS, Part 2 with ResNet18 trains in a few minutes per epoch.

## Training notes

- **OneCycleLR** scheduler (warmup + cosine annealing per batch) is used for
  the from-scratch trainings (AlexNet, CNN scratch). Tends to converge faster
  than `ReduceLROnPlateau`.
- **Label smoothing 0.1** in `CrossEntropyLoss` for the from-scratch parts to
  improve generalization.
- **Best-model checkpointing**: at the end of training, the weights are
  restored to the best-val epoch.
- **Early stopping**: training is aborted if val accuracy does not improve
  for N epochs (10 for CIFAR-10, 5 for X-Ray scratch, 4 for X-Ray transfer).
- **Validation split**: the original Chest X-Ray val set has only 16 images
  (useless for early stopping). We carve a real 10% val set out of train and
  append the original 16 images to test.
