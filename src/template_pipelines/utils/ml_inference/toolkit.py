"""Resuable methods and classes."""

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """Define a customized AutoEncoder model."""

    def __init__(self, n_dim):
        """Initialize AutoEncoder properties with defined layers and activation."""
        super().__init__()
        self.n_dim = n_dim

        # Encoder model
        self.encoder = nn.Sequential(
            nn.Linear(self.n_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )

        # Decoder model
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, self.n_dim)
        )

    def forward(self, x):
        """Define a model's forward pass to call the model on inputs and return outputs."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def convert_to_tensor(array):
    """Get a pytorch tensor from a numpy array."""
    return torch.tensor(array, dtype=torch.float32)
