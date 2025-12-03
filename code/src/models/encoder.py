# # import torch
# # import torch.nn as nn

# # class SFCEncoder(nn.Module):
# #     def __init__(self, input_dim=8, model_dim=128, num_heads=4, num_layers=3):
# #         super().__init__()

# #         self.embedding = nn.Linear(input_dim, model_dim)

# #         encoder_layer = nn.TransformerEncoderLayer(
# #             d_model=model_dim,
# #             nhead=num_heads,
# #             batch_first=True
# #         )

# #         self.transformer = nn.TransformerEncoder(
# #             encoder_layer,
# #             num_layers=num_layers
# #         )

# #     def forward(self, x):
# #         x = self.embedding(x)
# #         x = self.transformer(x)
# #         return torch.mean(x, dim=1)
# import torch
# import torch.nn as nn

# class Encoder(nn.Module):
#     def __init__(self, input_dim=10, hidden_dim=128):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
import torch
import torch.nn as nn

class SFCEncoder(nn.Module):
    def __init__(self, input_dim=8, model_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.mean(x, dim=1)
