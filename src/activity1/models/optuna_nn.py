import torch.nn as nn
 
class OptunaNN(nn.Module):
    """
    This class is dynamic. It builds the layers based on what 
    Optuna tells it to do. Every trial might have a different
    number of layers or neurons.
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
 
        layers = []
        current_dim = input_dim
 
        # Stack layers based on the hidden_dims list
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) # helps prevent overfitting
            current_dim = hidden_dim
 
        # Cap it off with the output layer
        layers.append(nn.Linear(current_dim, 1))
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
 
 
class FinalOptunaNN(nn.Module):
    """
    Basically the same as above, but we use this to build the winner
    once the tuning is over.
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
