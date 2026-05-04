# Deep Learning — Class Project

Repository containing the assignments for the Deep Learning course at
University of Deusto. Each activity is self-contained inside `src/`.

## Structure

```
src/
├── shared/              # utilities reused across activities
│   ├── logger.py        # setup_logger() + module-level logger
│   ├── seeds.py         # set_seeds()
│   └── utils.py         # save_model() / load_model()
│
├── activity1/           # Insurance charges — regression with NN
│   ├── README.md
│   ├── pipeline.py      # ShallowNN / DeepNN + Optuna tuning
│   └── ...
│
├── activity2/           # Computer vision — CNN from scratch + transfer learning
│   ├── README.md
│   ├── part1_cifar10/   # AlexNet adapted to CIFAR-10
│   └── part2_xray/      # CNN scratch vs ResNet18 transfer (Chest X-Ray pneumonia)
│
└── activity3/           # (TBD)
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Running

Each activity has its own entry point:

```bash
# Activity 1 — insurance regression
python -m src.activity1.main

# Activity 2 — Part 1: AlexNet on CIFAR-10
python -m src.activity2.part1_cifar10.main

# Activity 2 — Part 2: Chest X-Ray (CNN scratch + ResNet18 transfer)
python -m src.activity2.part2_xray.main

# Re-run Activity 2 without retraining (loads saved checkpoints)
SKIP_TRAINING=1 python -m src.activity2.part1_cifar10.main
SKIP_TRAINING=1 python -m src.activity2.part2_xray.main
```

See each activity's `README.md` for details on hyperparameters, datasets and
expected results.

## Hardware

Activities auto-detect the best PyTorch backend:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU
