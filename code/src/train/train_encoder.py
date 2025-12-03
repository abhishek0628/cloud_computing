# # import torch
# # import torch.optim as optim
# # from models.encoder import SFCEncoder

# # encoder = SFCEncoder()
# # optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

# # for epoch in range(50):
# #     sfc = torch.randn(16, 10, 8)
# #     embedding = encoder(sfc)
# #     loss = torch.mean(embedding ** 2)

# #     optimizer.zero_grad()
# #     loss.backward()
# #     optimizer.step()

# #     print(f"Encoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # torch.save(encoder.state_dict(), "encoder.pth")
# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from models.encoder import Encoder

# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='data/sfc_requests.npy')
# parser.add_argument('--save', type=str, default='checkpoints/encoder.pth')
# args = parser.parse_args()

# os.makedirs(os.path.dirname(args.save), exist_ok=True)

# # dummy dataset
# data = np.load(args.data) if os.path.exists(args.data) else np.random.rand(100,10).astype(np.float32)
# data = torch.tensor(data)

# encoder = Encoder(input_dim=data.shape[1])
# optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
# criterion = nn.MSELoss()

# for epoch in range(50):
#     optimizer.zero_grad()
#     output = encoder(data)
#     loss = criterion(output, data)  # example reconstruction loss
#     loss.backward()
#     optimizer.step()
#     print(f"Encoder Epoch {epoch} | Loss: {loss.item():.4f}")

# torch.save(encoder.state_dict(), args.save)
# print(f"Checkpoint saved at {args.save}")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from models.encoder import SFCEncoder

# ----------------- Arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/sfc_requests.npy')
parser.add_argument('--save', type=str, default='checkpoints/encoder.pth')
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

# ----------------- Load Data -----------------
sfc_requests = np.load(args.data)  # shape: (num_samples, seq_len, features)
data = torch.tensor(sfc_requests, dtype=torch.float32)

# ----------------- Model -----------------
encoder = SFCEncoder(input_dim=data.shape[2])
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # assuming regression task for training

# ----------------- Training -----------------
encoder.train()
for epoch in range(args.epochs):
    optimizer.zero_grad()
    output = encoder(data)
    loss = criterion(output, torch.zeros_like(output))  # dummy target
    loss.backward()
    optimizer.step()
    print(f"Encoder Epoch {epoch} | Loss: {loss.item():.4f}")

# ----------------- Save -----------------
torch.save(encoder.state_dict(), args.save)
print(f"Checkpoint saved at {args.save}")
