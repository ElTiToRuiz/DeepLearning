import optuna
import torch
import torch.nn as nn
import torch.optim as optim
 
from src.models.optuna_nn import OptunaNN
from src.logger.logger import logger
 
# Silence Optuna verbose output — use our own logger instead
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
 
def tune_with_optuna(X_train, y_train, X_test, y_test, input_dim, n_trials=20):
    """
    Search for the best hyperparameters using Optuna.
    Explores combinations of: number of layers, neurons per layer,
    learning rate, dropout, and weight decay.
    Returns the study with all trials and the best result.
    """
 
    def objective(trial):
 
        # ── HYPERPARAMETERS OPTUNA DECIDES IN EACH TRIAL ──────────────────
 
        # NUMBER OF HIDDEN LAYERS (between 1 and 3)
        n_layers = trial.suggest_int("n_layers", 1, 3)
 
        # NEURONS PER LAYER (each layer chosen independently)
        hidden_dims = [
            trial.suggest_int(f"hidden_dim_{i}", 16, 128)
            for i in range(n_layers)
        ]
 
        # LEARNING RATE in logarithmic scale (0.0001 to 0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
 
        # DROPOUT — regularization to avoid overfitting
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
 
        # WEIGHT DECAY — L2 regularization
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
 
        # EPOCHS
        epochs = trial.suggest_int("epochs", 100, 300)
 
        # ── MODEL ─────────────────────────────────────────────────────────
        model     = OptunaNN(input_dim, hidden_dims, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
        # ── TRAINING ──────────────────────────────────────────────────────
        for _ in range(epochs):
            model.train()
            predictions = model(X_train)
            loss        = criterion(predictions, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
        # ── METRIC THAT OPTUNA MINIMIZES ──────────────────────────────────
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss        = criterion(test_predictions, y_test)
 
        return test_loss.item()
 
    # CREATE THE STUDY AND RUN TRIALS
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
 
    # SHOW RESULTS
    best = study.best_trial
    logger.info(f"Optuna completed | Best test loss: {best.value:.4f}")
    logger.info(f"Best parameters: {best.params}")
 
    return study