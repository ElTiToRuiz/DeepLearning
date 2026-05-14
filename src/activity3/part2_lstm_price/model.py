"""
Generic LSTM classifier used by both part 2 (price-only) and part 3
(price + sentiment). Same architecture; only `n_features` changes.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, n_features)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * directions, B, hidden_dim)
        if self.bidirectional:
            # concat last layer forward + backward
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.head(self.dropout(last))
