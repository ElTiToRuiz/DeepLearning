import torch.nn as nn
 
class OptunaNN(nn.Module):
    """
    Dynamic architecture that builds the network according to the parameters
    passed by Optuna in each trial. The number of layers, neurons,
    and dropout_rate vary in each test.
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
 
        # CREATE THE NETWORK
        layers = []
        current_dim = input_dim
 
        # Iterate through the list of hidden layers chosen by Optuna
        for hidden_dim in hidden_dims:
            # LINEAR LAYER
            layers.append(nn.Linear(current_dim, hidden_dim))
 
            # RELU
            layers.append(nn.ReLU())
 
            # DROPOUT 
            layers.append(nn.Dropout(dropout_rate))
 
            # Update dimension for the next layer
            current_dim = hidden_dim
 
        # FINAL OUTPUT LAYER
        layers.append(nn.Linear(current_dim, 1)) # Single value for regression
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
 
 
# FINAL MODEL
class FinalOptunaNN(nn.Module):
    """
    Final model built with the best hyperparameters found by Optuna.
    Instantiated in pipeline.py after tuning with params from study.best_trial.
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
 
        layers = []
        current_dim = input_dim
 
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
 
        layers.append(nn.Linear(current_dim, 1))
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
