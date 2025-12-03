
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
