# Medical Insurance Cost Prediction — Neural Networks with PyTorch

A supervised regression project that trains and compares multiple neural network architectures to predict individual medical insurance costs based on demographic and health features. Built with PyTorch, scikit-learn, and Optuna for automated hyperparameter tuning.

---

## Authors

- **Mikel Sanchez**
- **Igor Ruiz**


---

## Project Structure

```
proyecto/
│
├── main.py                          # Entry point — runs the full pipeline
│
├── data/
│   └── insurance.csv                # Dataset
│
├── models/                          # Saved model weights (.pth)
├── results/                         # Saved plots and log files
│
└── src/
    ├── __init__.py
    ├── pipeline.py                  # Orchestrates the full training flow
    │
    ├── config/
    │   ├── __init__.py
    │   └── config.py                # All hyperparameters, paths and set_seeds()
    │
    ├── logger/
    │   ├── __init__.py
    │   └── logger.py                # Logging to console and timestamped file
    │
    ├── data/
    │   ├── __init__.py
    │   └── preprocess.py            # Data loading, encoding, scaling, tensor conversion
    │
    ├── models/
    │   ├── __init__.py
    │   ├── shallow_nn.py            # ShallowNN  — 1 hidden layer (input → 32 → 1)
    │   ├── deep_nn.py               # DeepNN     — 3 hidden layers (input → 64 → 32 → 16 → 1)
    │   └── optuna_nn.py             # OptunaNN + FinalOptunaNN — dynamic architecture
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train.py                 # Training loop (forward, backward, optimizer step)
    │   └── optuna_tuning.py         # Hyperparameter search with Optuna
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluate.py              # Metrics: MAE, RMSE, MedAE, MAPE, R²
    │   └── plots.py                 # Loss curves, Real vs Predicted, Residual analysis
    │
    └── utils/
        ├── __init__.py
        └── utils.py                 # save_model() and load_model()
```

---

## Pipeline

When you run `python main.py`, the following steps execute in order:

1. **Seeds** — All random seeds are fixed (Python, NumPy, PyTorch) to guarantee full reproducibility.
2. **Preprocessing** — The CSV is loaded, categorical features are one-hot encoded, data is split 80/20, features are standardized (fit on train only to avoid data leakage), and everything is converted to PyTorch tensors.
3. **ShallowNN** — A single hidden layer network is trained, saved, evaluated, and plotted.
4. **DeepNN** — A 3-layer funnel network is trained, saved, evaluated, and plotted.
5. **Comparison** — ShallowNN and DeepNN are compared side by side in a metrics table.
6. **Optuna Tuning** — Optuna runs N trials searching for the best combination of number of layers, neurons per layer, learning rate, dropout rate, weight decay, and number of epochs.
7. **FinalOptunaNN** — A model is built using the best parameters found by Optuna, trained from scratch, saved, and evaluated.
8. **Final Comparison** — All three models (ShallowNN, DeepNN, FinalOptunaNN) are compared in a single table.

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd proyecto

# Install dependencies
uv pip install -r requirements.txt
```

### Requirements

```
torch
pandas
scikit-learn
matplotlib
scipy
optuna
numpy
```

---

## Usage

```bash
python main.py
```

Trained model weights are saved to `models/`. All plots and log files are saved to `results/`. Each run generates a new timestamped log file so previous runs are never overwritten.

---

## Reproducibility

All random seeds are fixed at the start of every run via `set_seeds()` in `config.py`. This covers Python's `random` module, NumPy, and PyTorch (both CPU and GPU). Running `python main.py` will always produce the same results.