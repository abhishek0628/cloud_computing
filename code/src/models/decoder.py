
import torch
import torch.nn as nn

class SFCDecoder(nn.Module):
    def __init__(self, model_dim=128, num_heads=4, num_layers=3):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_fc = nn.Linear(model_dim, model_dim)  # keep output in model_dim

    def forward(self, x, memory):
        """
        x: (batch, seq_len, model_dim)
        memory: (batch, seq_len, model_dim)
        """
        out = self.transformer(x, memory)
        out = self.output_fc(out)
        return torch.softmax(out, dim=-1)
