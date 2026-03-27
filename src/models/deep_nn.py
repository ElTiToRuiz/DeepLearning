import torch.nn as nn 
 
class DeepNN(nn.Module):
    """
    Deep neural network with 3 hidden layers.
    Funnel architecture: input → 64 → 32 → 16 → 1
    Each layer learns more abstract representations.
    """
    def __init__(self, input_dim):
        super().__init__()
 
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
 
            nn.Linear(64, 32),
            nn.ReLU(),
 
            nn.Linear(32, 16),
            nn.ReLU(),
 
            nn.Linear(16, 1)  # output: 1 value (regression)
        )
 
    def forward(self, x):
        return self.network(x)