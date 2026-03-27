import optuna
import torch
import torch.nn as nn
import torch.optim as optim
 
from src.models.optuna_nn import OptunaNN
from src.logger.logger import logger
 
# Turn down Optuna's noise since we'll use our own logger for the important bits
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
 
def tune_with_optuna(X_train, y_train, X_test, y_test, input_dim, n_trials=20):
    """
    Fire up Optuna to find the best settings for the model.
    It'll mess around with depth, width, and some weight settings.
    Returns the study object with all the results.
    """
 
    def objective(trial):
 
        # -- Parameters that Optuna will tweak --
 
        # Number of layers to stack (from 1 to 3)
        n_layers = trial.suggest_int("n_layers", 1, 3)
 
        # How many neurons in each layer? (picked per layer)
        hidden_dims = [
            trial.suggest_int(f"hidden_dim_{i}", 16, 128)
            for i in range(n_layers)
        ]
 
        # Learning rate search range (using a log scale makes more sense here)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
 
        # Dropout to help with overfitting
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
 
        # L2 Regularization (weight decay)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
 
        # Epoch count for the trial
        epochs = trial.suggest_int("epochs", 100, 300)
 
        # -- Set up the trial model --
        model     = OptunaNN(input_dim, hidden_dims, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
        # -- Training loop for this specific trial --
        for _ in range(epochs):
            model.train()
            predictions = model(X_train)
            loss        = criterion(predictions, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
        # -- Check how it did on the test set --
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss        = criterion(test_predictions, y_test)
 
        return test_loss.item()
 
    # Create the study and start the search
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
 
    # Log what was found
    best = study.best_trial
    logger.info(f"Done with Optuna trials | Lowest test loss: {best.value:.4f}")
    logger.info(f"Top params found: {best.params}")
 
    return study