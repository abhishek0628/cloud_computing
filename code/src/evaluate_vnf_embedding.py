import torch
import numpy as np
import matplotlib.pyplot as plt
from models.encoder import SFCEncoder

# Load encoder
encoder = SFCEncoder()
encoder.load_state_dict(torch.load("checkpoints/encoder.pth"))
encoder.eval()

# Load SFC requests
sfc_requests = torch.tensor(np.load("data/sfc_requests.npy"), dtype=torch.float)

embedding_costs = []

for i in range(len(sfc_requests)):
    with torch.no_grad():
        embed = encoder(sfc_requests[i:i+1])  # (1, seq_len, model_dim)
        cost = torch.norm(embed)  # simple L2 norm as "embedding cost"
        embedding_costs.append(cost.item())

# Plot convergence
plt.plot(embedding_costs)
plt.xlabel("SFC Index")
plt.ylabel("Embedding Cost (L2 norm)")
plt.title("VNF Embedding Cost Convergence")
plt.savefig("vnf_embedding_convergence.png")
plt.show()
