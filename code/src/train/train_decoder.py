# # import torch
# # import torch.optim as optim
# # from models.decoder import SFCDecoder
# # from models.encoder import SFCEncoder
# # import numpy as np

# # encoder = SFCEncoder()
# # encoder.load_state_dict(torch.load("encoder.pth"))
# # decoder = SFCDecoder()

# # optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

# # for epoch in range(100):

# #     sfc = torch.randn(16, 10, 8)
# #     sfc_embed = encoder(sfc)

# #     probs = decoder(sfc_embed)
# #     actions = torch.multinomial(probs, 1)

# #     rewards = -torch.var(actions.float())

# #     loss = -torch.mean(torch.log(probs.gather(1, actions)) * rewards)

# #     optimizer.zero_grad()
# #     loss.backward()
# #     optimizer.step()

# #     print(f"Decoder Epoch {epoch} | Reward: {rewards.item():.4f}")
# # # # # # import argparse
# # # # # # import os
# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.optim as optim
# # # # # # import numpy as np
# # # # # # from models.decoder import SFCDecoder

# # # # # # parser = argparse.ArgumentParser()
# # # # # # parser.add_argument('--data', type=str, default='data/sfc_requests.npy')
# # # # # # parser.add_argument('--save', type=str, default='checkpoints/decoder.pth')
# # # # # # args = parser.parse_args()

# # # # # # os.makedirs(os.path.dirname(args.save), exist_ok=True)

# # # # # # data = np.load(args.data) if os.path.exists(args.data) else np.random.rand(100,128).astype(np.float32)
# # # # # # data = torch.tensor(data)
# # # # # # target = torch.randn(data.shape[0], 1)  # dummy targets

# # # # # # decoder = SFCDecoder(input_dim=data.shape[1])
# # # # # # optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
# # # # # # criterion = nn.MSELoss()

# # # # # # for epoch in range(100):
# # # # # #     optimizer.zero_grad()
# # # # # #     output = decoder(data)
# # # # # #     loss = criterion(output, target)
# # # # # #     loss.backward()
# # # # # #     optimizer.step()
# # # # # #     print(f"Decoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # # # torch.save(decoder.state_dict(), args.save)
# # # # # # print(f"Checkpoint saved at {args.save}")
# # # # # import os
# # # # # import argparse
# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from models.decoder import SFCDecoder  # your transformer decoder class
# # # # # from models.encoder import SFCEncoder  # encoder to provide memory

# # # # # # -------------------------------
# # # # # # Argument parser
# # # # # # -------------------------------
# # # # # parser = argparse.ArgumentParser()
# # # # # parser.add_argument('--data', type=str, default='data/sfc_requests.npy')
# # # # # parser.add_argument('--save', type=str, default='checkpoints/decoder.pth')
# # # # # parser.add_argument('--epochs', type=int, default=100)
# # # # # parser.add_argument('--lr', type=float, default=1e-4)
# # # # # args = parser.parse_args()

# # # # # # -------------------------------
# # # # # # Load SFC request data
# # # # # # -------------------------------
# # # # # if not os.path.exists(args.data):
# # # # #     raise FileNotFoundError(f"{args.data} not found. Run generate_sfc_requests.py first.")

# # # # # sfc_requests = np.load(args.data)  # shape: (num_samples, seq_len, features)
# # # # # sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)

# # # # # # -------------------------------
# # # # # # Initialize models
# # # # # # -------------------------------
# # # # # encoder = SFCEncoder()
# # # # # decoder = SFCDecoder()
# # # # # criterion = nn.MSELoss()
# # # # # optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

# # # # # # Compute encoder memory
# # # # # with torch.no_grad():
# # # # #     memory = encoder(sfc_requests)  # shape: (num_samples, model_dim)

# # # # # # -------------------------------
# # # # # # Training loop
# # # # # # -------------------------------
# # # # # for epoch in range(args.epochs):
# # # # #     decoder.train()
# # # # #     optimizer.zero_grad()
    
# # # # #     # Forward pass: pass both input and encoder memory
# # # # #     output = decoder(sfc_requests, memory)
    
# # # # #     # Dummy target: for real tasks replace with actual SFC output
# # # # #     target = torch.zeros_like(output)
    
# # # # #     loss = criterion(output, target)
# # # # #     loss.backward()
# # # # #     optimizer.step()
    
# # # # #     print(f"Decoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # # # -------------------------------
# # # # # # Save checkpoint
# # # # # # -------------------------------
# # # # # os.makedirs(os.path.dirname(args.save), exist_ok=True)
# # # # # torch.save(decoder.state_dict(), args.save)
# # # # # print(f"Checkpoint saved at {args.save}")
# # # # import argparse
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # import numpy as np
# # # # from models.decoder import SFCDecoder  # make sure this matches your file

# # # # # ------------------------------
# # # # # Argument parser
# # # # # ------------------------------
# # # # parser = argparse.ArgumentParser()
# # # # parser.add_argument("--data", type=str, required=True, help="Path to SFC requests numpy file")
# # # # parser.add_argument("--save", type=str, required=True, help="Path to save decoder checkpoint")
# # # # parser.add_argument("--epochs", type=int, default=100)
# # # # parser.add_argument("--lr", type=float, default=1e-4)
# # # # args = parser.parse_args()

# # # # # ------------------------------
# # # # # Load data
# # # # # ------------------------------
# # # # sfc_requests = np.load(args.data)  # shape: (num_samples, seq_len, features)
# # # # sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)

# # # # # ------------------------------
# # # # # Initialize decoder
# # # # # ------------------------------
# # # # decoder = SFCDecoder()
# # # # optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
# # # # loss_fn = nn.MSELoss()  # example loss; adjust for your task

# # # # # ------------------------------
# # # # # Training loop
# # # # # ------------------------------
# # # # for epoch in range(args.epochs):
# # # #     optimizer.zero_grad()

# # # #     # Forward pass
# # # #     # For TransformerDecoder, query and memory must be 3D: (batch, seq_len, features)
# # # #     # Here we use sfc_requests as both query and "memory"
# # # #     output = decoder(sfc_requests, sfc_requests)

# # # #     # Target: can be the same as input for auto-regressive training
# # # #     target = sfc_requests
# # # #     loss = loss_fn(output, target)

# # # #     loss.backward()
# # # #     optimizer.step()

# # # #     print(f"Decoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # # ------------------------------
# # # # # Save checkpoint
# # # # # ------------------------------
# # # # torch.save(decoder.state_dict(), args.save)
# # # # print(f"Decoder checkpoint saved at {args.save}")
# # # # 
# # # import argparse
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # import numpy as np
# # # from models.decoder import SFCDecoder

# # # # ------------------------------
# # # # Arguments
# # # # ------------------------------
# # # parser = argparse.ArgumentParser()
# # # parser.add_argument("--data", type=str, required=True)
# # # parser.add_argument("--save", type=str, required=True)
# # # parser.add_argument("--epochs", type=int, default=50)
# # # parser.add_argument("--lr", type=float, default=1e-4)
# # # parser.add_argument("--input_dim", type=int, default=8)
# # # parser.add_argument("--model_dim", type=int, default=128)
# # # args = parser.parse_args()

# # # # ------------------------------
# # # # Load data
# # # # ------------------------------
# # # sfc_requests = np.load(args.data)  # (num_samples, seq_len, input_dim)
# # # sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)
# # # num_samples, seq_len, input_dim = sfc_requests.shape

# # # # ------------------------------
# # # # Model & embedding
# # # # ------------------------------
# # # decoder = SFCDecoder(input_dim=args.input_dim, model_dim=args.model_dim)
# # # optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
# # # loss_fn = nn.MSELoss()

# # # # Input embedding to match model_dim
# # # embedding = nn.Linear(args.input_dim, args.model_dim)

# # # # ------------------------------
# # # # Training loop
# # # # ------------------------------
# # # for epoch in range(args.epochs):
# # #     optimizer.zero_grad()

# # #     # Embed input and transpose to (seq_len, batch, model_dim)
# # #     x = embedding(sfc_requests)             # (batch, seq_len, model_dim)
# # #     x = x.transpose(0, 1)                   # (seq_len, batch, model_dim)
# # #     memory = x.clone()                       # use same as memory

# # #     # Forward pass
# # #     output = decoder(x, memory)             # (seq_len, batch, model_dim)

# # #     # Target: same shape as output
# # #     target = memory
# # #     loss = loss_fn(output, target)

# # #     loss.backward()
# # #     optimizer.step()

# # #     print(f"Decoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # ------------------------------
# # # # Save checkpoint
# # # # ------------------------------
# # # torch.save(decoder.state_dict(), args.save)
# # # print(f"Decoder checkpoint saved at {args.save}")
# # # import argparse
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # import numpy as np
# # # from models.decoder import SFCDecoder  # decoder should have its own embedding

# # # # ------------------------------
# # # # Arguments
# # # # ------------------------------
# # # parser = argparse.ArgumentParser()
# # # parser.add_argument("--data", type=str, required=True)
# # # parser.add_argument("--save", type=str, required=True)
# # # parser.add_argument("--epochs", type=int, default=50)
# # # parser.add_argument("--lr", type=float, default=1e-4)
# # # parser.add_argument("--input_dim", type=int, default=8)
# # # parser.add_argument("--model_dim", type=int, default=128)
# # # args = parser.parse_args()

# # # # ------------------------------
# # # # Load data
# # # # ------------------------------
# # # sfc_requests = np.load(args.data)  # (num_samples, seq_len, input_dim)
# # # sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)
# # # num_samples, seq_len, input_dim = sfc_requests.shape

# # # # ------------------------------
# # # # Model
# # # # ------------------------------
# # # decoder = SFCDecoder(input_dim=args.input_dim, model_dim=args.model_dim)
# # # optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
# # # loss_fn = nn.MSELoss()

# # # # ------------------------------
# # # # Training loop
# # # # ------------------------------
# # # for epoch in range(args.epochs):
# # #     optimizer.zero_grad()

# # #     # Transpose to (seq_len, batch, input_dim)
# # #     x = sfc_requests.transpose(0, 1)  # (seq_len, batch, input_dim)
# # #     memory = x.clone()                 # use same as memory

# # #     # Forward pass through decoder
# # #     output = decoder(x, memory)        # decoder handles its own embedding

# # #     # Target: same as memory after decoder embedding
# # #     # If decoder internally embeds, then output and memory shapes match
# # #     target = output.detach()           # simple self-supervised example
# # #     loss = loss_fn(output, target)

# # #     loss.backward()
# # #     optimizer.step()

# # #     print(f"Decoder Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # ------------------------------
# # # # Save checkpoint
# # # # ------------------------------
# # # torch.save(decoder.state_dict(), args.save)
# # # print(f"Decoder checkpoint saved at {args.save}")
# # src/train/train_decoder.py
# import torch
# import torch.optim as optim
# import numpy as np
# from models.decoder import SFCDecoder
# from models.encoder import SFCEncoder
# import argparse
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--data", type=str, default="data/sfc_requests.npy")
# parser.add_argument("--save", type=str, default="checkpoints/decoder.pth")
# args = parser.parse_args()

# # Load SFC requests
# sfc_requests = np.load(args.data)  # shape: (num_samples, seq_len, features)
# sfc_requests = torch.tensor(sfc_requests, dtype=torch.float)

# # Initialize encoder and decoder
# encoder = SFCEncoder()
# encoder_ckpt = "checkpoints/encoder.pth"
# encoder.load_state_dict(torch.load(encoder_ckpt))
# encoder.eval()  # set encoder to eval mode

# decoder = SFCDecoder()
# optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
# loss_fn = torch.nn.MSELoss()

# # Training loop
# for epoch in range(50):
#     total_loss = 0.0
#     for batch_idx in range(0, sfc_requests.shape[0], 16):
#         batch = sfc_requests[batch_idx: batch_idx + 16]  # (batch, seq_len, features)
#         memory = encoder(batch)  # (batch, model_dim)
#         memory = memory.unsqueeze(1).repeat(1, batch.shape[1], 1)  # expand to seq_len

#         optimizer.zero_grad()
#         output = decoder(batch, memory)  # feed batch and memory
#         loss = loss_fn(output, batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# # Save decoder checkpoint
# os.makedirs(os.path.dirname(args.save), exist_ok=True)
# torch.save(decoder.state_dict(), args.save)
# print(f"Decoder checkpoint saved at {args.save}")
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
