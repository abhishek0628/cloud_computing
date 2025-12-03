# # # import torch
# # # import torch.nn as nn

# # # class VNFDecoder(nn.Module):
# # #     def __init__(self, embed_dim=128, num_nodes=50):
# # #         super().__init__()

# # #         self.fc1 = nn.Linear(embed_dim, 256)
# # #         self.fc2 = nn.Linear(256, num_nodes)

# # #     def forward(self, sfc_embedding):
# # #         x = torch.relu(self.fc1(sfc_embedding))
# # #         logits = self.fc2(x)
# # #         probs = torch.softmax(logits, dim=-1)
# # #         return probs
# # import torch
# # import torch.nn as nn

# # class SFCDecoder(nn.Module):
# #     def __init__(self, input_dim=8, model_dim=128, num_heads=4, num_layers=3):
# #         super().__init__()

# #         self.embedding = nn.Linear(input_dim, model_dim)

# #         decoder_layer = nn.TransformerDecoderLayer(
# #             d_model=model_dim,
# #             nhead=num_heads,
# #             batch_first=True
# #         )

# #         self.transformer = nn.TransformerDecoder(
# #             decoder_layer,
# #             num_layers=num_layers
# #         )

# #     def forward(self, x, memory):
# #         x = self.embedding(x)
# #         x = self.transformer(x, memory)
# #         return torch.mean(x, dim=1)
# # src/models/decoder.py
# import torch
# import torch.nn as nn

# class SFCDecoder(nn.Module):
#     def __init__(self, input_dim=128, model_dim=128, num_heads=4, num_layers=3, seq_len=10):
#         super().__init__()
#         self.embedding = nn.Linear(input_dim, model_dim)  # project input to model_dim

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=model_dim,
#             nhead=num_heads,
#             batch_first=True
#         )

#         self.transformer = nn.TransformerDecoder(
#             decoder_layer,
#             num_layers=num_layers
#         )

#         self.output_fc = nn.Linear(model_dim, input_dim)  # output same feature size

#     def forward(self, x, memory):
#         """
#         x: (batch, seq_len, input_dim)
#         memory: (batch, seq_len, model_dim)
#         """
#         x = self.embedding(x)        # (batch, seq_len, model_dim)
#         out = self.transformer(x, memory)  # standard Transformer decoder
#         out = self.output_fc(out)    # map back to original feature dim
#         return torch.softmax(out, dim=-1)
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
