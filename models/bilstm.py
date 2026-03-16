import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    """
    Improved Bidirectional LSTM for IMDB sentiment classification.

    Changes over v1:
      - 2 stacked BiLSTM layers (num_layers=2) for deeper sequence modelling
      - LayerNorm applied to the final hidden state for training stability
      - Two-layer FC head (hidden→hidden//2→1) for richer representation
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_units=128,
                 dropout=0.3, num_layers=2):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_units * 2   # bidirectional

        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout    = nn.Dropout(dropout)

        # Two-layer FC head
        self.fc1 = nn.Linear(lstm_out_dim, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.embedding(x)                           # (B, T, E)
        _, (hidden, _) = self.lstm(x)                  # hidden: (2*layers, B, H)

        # Concatenate the last layer's forward + backward final hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)   # (B, 2H)

        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)

        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return torch.sigmoid(out)
