# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from models.decoder import SFCDecoder

# # Load decoder
# decoder = SFCDecoder()
# decoder.load_state_dict(torch.load("checkpoints/decoder.pth"))
# decoder.eval()

# # Load SFC requests
# sfc_requests = torch.tensor(np.load("data/sfc_requests.npy"), dtype=torch.float)

# deployment_costs = []

# for i in range(len(sfc_requests)):
#     with torch.no_grad():
#         memory = torch.randn(1, sfc_requests.shape[2], 128)  # dummy memory for transformer decoder
#         out = decoder(sfc_requests[i:i+1], memory)
#         cost = -torch.max(out)  # simple placeholder for deployment cost
#         deployment_costs.append(cost.item())

# # Plot convergence
# plt.plot(deployment_costs)
# plt.xlabel("SFC Index")
# plt.ylabel("Deployment Cost")
# plt.title("SFC Deployment Cost Convergence")
# plt.savefig("sfc_deployment_convergence.png")
# plt.show()
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.encoder import SFCEncoder
from models.decoder import SFCDecoder

# Load models
encoder = SFCEncoder()
encoder.load_state_dict(torch.load("checkpoints/encoder.pth"))
encoder.eval()

decoder = SFCDecoder()
decoder.load_state_dict(torch.load("checkpoints/decoder.pth"))
decoder.eval()

# Load SFC requests
sfc_requests = torch.tensor(np.load("data/sfc_requests.npy"), dtype=torch.float)

deployment_costs = []

for i in range(len(sfc_requests)):
    with torch.no_grad():
        # First embed with encoder
        memory = encoder(sfc_requests[i:i+1])  # (1, seq_len, 128)
        # Then decode
        out = decoder(memory, memory)  # decoder expects (seq_len, batch, model_dim)
        # Example: simple deployment cost = sum over decoder output
        cost = out.sum().item()
        deployment_costs.append(cost)

# Plot convergence
plt.plot(deployment_costs)
plt.xlabel("SFC Index")
plt.ylabel("Deployment Cost")
plt.title("SFC Deployment Cost Convergence")
plt.savefig("sfc_deployment_convergence.png")
plt.show()
