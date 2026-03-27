import torch.nn as nn 
 
class DeepNN(nn.Module):
    """
    A bit more complex, using 3 hidden layers.
    It funnels down from 64 to 32 to 16, then finally to the output.
    Helps the model pick up on more subtle patterns.
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
 
            nn.Linear(16, 1)  # Final regression output
        )
 
    def forward(self, x):
        return self.network(x)