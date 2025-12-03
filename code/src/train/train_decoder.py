
import torch
import torch.optim as optim
from models.decoder import SFCDecoder
from models.encoder import SFCEncoder
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/sfc_requests.npy")
parser.add_argument("--save", type=str, default="checkpoints/decoder.pth")
args = parser.parse_args()

# Load SFC requests
sfc_requests = np.load(args.data)
sfc_requests = torch.tensor(sfc_requests, dtype=torch.float)

# Encoder
encoder = SFCEncoder()
encoder.load_state_dict(torch.load("checkpoints/encoder.pth"))
encoder.eval()  # do not train encoder

# Decoder
decoder = SFCDecoder()
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(50):
    total_loss = 0.0
    for batch_idx in range(0, sfc_requests.shape[0], 16):
        batch = sfc_requests[batch_idx: batch_idx + 16]
        memory = encoder(batch)  # (batch, model_dim)
        memory = memory.unsqueeze(1).repeat(1, batch.shape[1], 1)  # expand to seq_len
        decoder_input = memory  # use memory as input (query)

        optimizer.zero_grad()
        output = decoder(decoder_input, memory)  # query and memory
        loss = loss_fn(output, memory)  # compare to memory (embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# Save decoder
os.makedirs(os.path.dirname(args.save), exist_ok=True)
torch.save(decoder.state_dict(), args.save)
print(f"Decoder checkpoint saved at {args.save}")
