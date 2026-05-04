import torch.nn as nn
  
class ShallowNN(nn.Module):
    """
    Very basic setup with just one hidden layer.
    Input -> 32 nodes -> Output
    Small and fast, perfect as a starting point.
    """
    def __init__(self, input_dim):
        super().__init__()
 
        # Hidden layer: take input and map to 32 neurons
        self.hidden = nn.Linear(input_dim, 32)
 
        # ReLU to keep things non-linear
        self.relu = nn.ReLU()
 
        # Just one output node for our price prediction
        self.output = nn.Linear(32, 1)
 
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)