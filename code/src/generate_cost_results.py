import torch
import numpy as np
from models.encoder import SFCEncoder
from models.decoder import SFCDecoder

# Load models
encoder = SFCEncoder()
encoder.load_state_dict(torch.load("encoder.pth"))
encoder.eval()

decoder = SFCDecoder()
decoder.load_state_dict(torch.load("decoder.pth"))
decoder.eval()

# Load or generate SFC requests
sfc_requests = np.load("data/sfc_requests.npy")  # shape: (num_requests, seq_len, features)
sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)

nn_costs = []

with torch.no_grad():
    for i in range(len(sfc_requests)):
        sfc = sfc_requests[i:i+1]  # single request
        memory = encoder(sfc)
        probs = decoder(sfc, memory)
        # Example cost: negative log likelihood as proxy
        cost = -torch.mean(torch.log(probs + 1e-8)).item()
        nn_costs.append(cost)

# Save results
nn_costs = np.array(nn_costs)
np.save("results/nn_costs.npy", nn_costs)
print("Saved nn_costs.npy")
