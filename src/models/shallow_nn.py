import torch.nn as nn
  
class ShallowNN(nn.Module):
    """
    Neural network with a single hidden layer.
    Architecture: input → 32 → 1
    Simple, fast, good as a baseline.
    """
    def __init__(self, input_dim):
        super().__init__()
 
        # HIDDEN LAYER: input_dim → 32 neurons
        self.hidden = nn.Linear(input_dim, 32)
 
        # ReLU — introduces non-linearity to learn complex patterns
        self.relu = nn.ReLU()
 
        # OUTPUT LAYER: 1 neuron for regression (predicting a value)
        self.output = nn.Linear(32, 1)
 
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x