# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from . import config


#GRU sobre la curva de luz. Recibe la secuencia con el padding y el vector con la longitud real para ignorar el relleno en el entrenamiento.
class PhotometryGRUEncoder(nn.Module):

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64,
                 num_layers: int = 1, embedding_dim: int = config.EMBEDDING_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, photometry: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # photometry: (B, T_max, input_dim); lengths: (B,)
        packed = pack_padded_sequence(
            photometry, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        last_layer_hidden = h_n[-1]                # (B, hidden_dim)
        embedding = self.norm(self.proj(last_layer_hidden))
        return embedding                            # (B, embedding_dim)


#CNN 1D para espectro.
class SpectraCNNEncoder(nn.Module):

    def __init__(self, in_channels: int = 3, embedding_dim: int = config.EMBEDDING_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(64, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        features = self.conv(spectra).squeeze(-1)
        features = self.dropout(features)
        embedding = self.norm(self.proj(features))
        return embedding


#MLP para metadata
class MetadataMLPEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 embedding_dim: int = config.EMBEDDING_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        features = self.net(metadata)
        embedding = self.norm(self.proj(features))
        return embedding

#design pattern que envuelve todos los encoders igual. Devuelve los logits y el embedding
class UnimodalClassifier(nn.Module):

    def __init__(self, encoder: nn.Module, embedding_dim: int = config.EMBEDDING_DIM,
                 num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, *args, **kwargs):
        embedding = self.encoder(*args, **kwargs)
        logits = self.head(embedding)
        return logits, embedding
