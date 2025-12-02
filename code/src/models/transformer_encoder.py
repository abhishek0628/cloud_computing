"""
Transformer encoder for encoding VNF resource vectors.

API:
    model = TransformerEncoderModel(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
    # input: vnfs: Tensor of shape (batch_size, seq_len, feature_dim) OR (seq_len, feature_dim)
    enc_out = model(vnfs)  # returns Tensor (batch_size, seq_len, d_model) or (seq_len, d_model)

Notes:
 - Uses a linear projection from raw VNF features -> d_model
 - Adds sinusoidal positional encodings
 - Uses torch.nn.TransformerEncoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd case: pad the cos column
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: x + positional encoding (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                 feature_dim: int = 5,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 max_len: int = 128):
        """
        feature_dim: dimension of raw VNF feature vector (e.g., cpu, ram, bw, energy, dur)
        d_model: model hidden size
        nhead: attention heads
        num_layers: number of Transformer encoder layers
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # input projection: raw features -> d_model
        self.input_fc = nn.Linear(feature_dim, d_model)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)  # batch_first=True to accept (batch, seq, d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # optional output layer (identity for now)
        self.output_fc = nn.Identity()

        # initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_fc.weight)
        if self.input_fc.bias is not None:
            nn.init.zeros_(self.input_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: either
            - torch.Tensor shape (seq_len, feature_dim)
            - torch.Tensor shape (batch_size, seq_len, feature_dim)

        returns:
            enc_out: torch.Tensor shape (batch_size, seq_len, d_model) OR (seq_len, d_model)
        """
        single_sequence = False

        if x.dim() == 2:
            # (seq_len, feature_dim) -> add batch dim
            x = x.unsqueeze(0)
            single_sequence = True
        elif x.dim() == 3:
            # (batch_size, seq_len, feature_dim) -> ok
            pass
        else:
            raise ValueError("Input tensor must be 2D or 3D: (seq_len, feature_dim) or (batch, seq_len, feature_dim)")

        # project -> (batch, seq_len, d_model)
        x = self.input_fc(x)
        x = self.pos_enc(x)          # add positional encoding
        enc_out = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        enc_out = self.output_fc(enc_out)

        if single_sequence:
            return enc_out.squeeze(0)  # (seq_len, d_model)
        return enc_out  # (batch, seq_len, d_model)


if __name__ == "__main__":
    # quick unit test
    model = TransformerEncoderModel(feature_dim=5, d_model=32, nhead=4, num_layers=2)
    sample = torch.randn(3, 5)            # seq_len=3, feature_dim=5
    out = model(sample)                   # output (3, 32)
    print("Out shape (single):", out.shape)
    batch = torch.randn(4, 6, 5)          # batch=4, seq_len=6, feature_dim=5
    out2 = model(batch)                   # (4,6,32)
    print("Out shape (batch):", out2.shape)
