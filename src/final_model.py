import torch
import torch.nn as nn

# MODELO FINAL 
class FinalOptunaNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 95),
            nn.ReLU(),
            nn.Dropout(0.36047474490948217),

            nn.Linear(95, 79),
            nn.ReLU(),
            nn.Dropout(0.36047474490948217),

            nn.Linear(79, 127),
            nn.ReLU(),
            nn.Dropout(0.36047474490948217),

            nn.Linear(127, 1)
        )

    def forward(self, x):
        return self.network(x)