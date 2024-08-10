import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TaggerTransformer(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048):
        super(TaggerTransformer, self).__init__()
        # Embedding layer for position and patch embedding
        self.embedding = nn.Conv2d(3, d_model, kernel_size=16, stride=16)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Create patch embeddings
        x = self.embedding(x)  # Output shape: (batch_size, d_model, grid_size, grid_size)
        x = x.flatten(2)  # Flatten grid
        x = x.permute(2, 0, 1)  # Change to (seq_len, batch, feature) for the transformer

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use only the first token for classification or mean pooling
        x = x.mean(dim=0)  # Mean pooling over sequence

        # Classifier
        x = self.classifier(x)
        return x
