# Predicting Medical Insurance Costs — Pytorch Neural Networks

This project is all about using neural networks to predict how much medical insurance is going to cost. We've built and compared a few different architectures using PyTorch and threw in some Optuna for automated hyperparameter tuning to find the best settings.

---

## Authors

- **Mikel Sanchez**
- **Igor Ruiz**

---

## Project Structure

Here's how we've organized everything:

```
DeepLearning/
│
├── main.py                          # The entry point to run the whole thing
│
├── data/
│   └── insurance.csv                # Our dataset
│
├── models/                          # Where we save the trained weights (.pth)
├── results/                         # Where the plots and logs end up
│
└── src/
    ├── __init__.py
    ├── pipeline.py                  # The main script orchestrating everything
    │
    ├── config/
    │   └── config.py                # All our knobs, paths, and seed settings
    │
    ├── logger/
    │   └── logger.py                # Handles console and file logging
    │
    ├── data/
    │   └── preprocess.py            # Clean, encode, and scale the data into tensors
    │
    ├── models/
    │   ├── shallow_nn.py            # Basic network (1 hidden layer)
    │   ├── deep_nn.py               # More complex network (3 layers)
    │   └── optuna_nn.py             # Dynamic model for the Optuna search
    │
    ├── training/
    │   ├── train.py                 # The standard training loop logic
    │   └── optuna_tuning.py         # The logic behind the hyperparameter search
    │
    ├── evaluation/
    │   ├── evaluate.py              # Calculating MAE, RMSE, R², etc.
    │   └── plots.py                 # Generating all the pretty charts
    │
    └── utils/
        └── utils.py                 # Simple helpers to save and load models
```

---

## How the Pipeline Works

When you fire off `python main.py`, it goes through these steps:

1.  **Seeds**: We lock all random seeds (Python, NumPy, PyTorch) so the results are actually reproducible.
2.  **Preprocessing**: Load the CSV, one-hot encode categories, split 80/20, and scale the features. We only "fit" the scaler on the train set to avoid any data leakage.
3.  **ShallowNN**: Train a basic model with one hidden layer just to see where we stand.
4.  **DeepNN**: Step it up with a 3-layer network that funnels down from 64 to 16 neurons.
5.  **Initial Comparison**: See how the Shallow and Deep models stack up against each other.
6.  **Optuna Tuning**: Let Optuna run 20 trials to hunt down the best depth, width, learning rate, and dropout settings.
7.  **Final Trial**: Take the absolute best parameters found by Optuna and train one last "Final" model.
8.  **Final Comparison**: One last table to compare all three approaches (Shallow, Deep, and the Optuna winner).

---

## Setup

We used [uv](https://github.com/astral-sh/uv) to manage dependencies.

```bash
# Clone the repo
git clone <repo-url>
cd DeepLearning

# Install the goods
uv pip install -r requirements.txt
```

### Requirements

- `torch`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `optuna`
- `numpy`

---

## Running It

```bash
python main.py
```

Weights go into `models/`, and your charts and logs land in `results/`. Every time you run it, a new timestamped log is created so you won't lose your previous results.

---

## Consistency

Since we fixed the seeds in `config.py`, you should get the same results every single time you run `python main.py`. No magic, just reproducible math.