
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.encoder import SFCEncoder
from models.decoder import SFCDecoder
from models.gnn_hueristic import GNNHeuristic

# --------------------------
# 1. Load checkpoints
# --------------------------
encoder = SFCEncoder()
encoder.load_state_dict(torch.load("encoder.pth"))
encoder.eval()

decoder = SFCDecoder()
decoder.load_state_dict(torch.load("decoder.pth"))
decoder.eval()

gnn = GNNHeuristic()
gnn.load_state_dict(torch.load("checkpoints/gnn_heuristic.pth"))
gnn.eval()

# --------------------------
# 2. Load SFC requests
# --------------------------
sfc_requests = np.load("data/sfc_requests.npy")  # shape: [num_samples, seq_len, 8]
sfc_requests = torch.tensor(sfc_requests, dtype=torch.float32)

num_samples = sfc_requests.shape[0]

# --------------------------
# 3. Compute VNF Embedding Cost Convergence
# --------------------------
vnf_embedding_costs = []

for i in range(num_samples):
    sfc_sample = sfc_requests[i:i+1]          # [1, seq_len, 8]
    memory = encoder(sfc_sample)              # [1, seq_len, 128]
    probs = decoder(sfc_sample, memory)       # decoder expects memory
    cost = -torch.sum(probs).item()           # just an example cost metric
    vnf_embedding_costs.append(cost)

plt.plot(vnf_embedding_costs)
plt.xlabel("SFC sample index")
plt.ylabel("VNF embedding cost")
plt.title("Convergence of VNF Embedding Cost")
plt.savefig("vnf_embedding_convergence.png")
plt.close()

# --------------------------
# 4. Compute SFC Deployment Cost Convergence
# --------------------------
sfc_deployment_costs = []

for i in range(num_samples):
    sfc_sample = sfc_requests[i:i+1]
    memory = encoder(sfc_sample)
    probs = decoder(sfc_sample, memory)
    deployment_cost = -torch.sum(probs).item()  # dummy deployment cost
    sfc_deployment_costs.append(deployment_cost)

plt.plot(sfc_deployment_costs)
plt.xlabel("SFC sample index")
plt.ylabel("SFC deployment cost")
plt.title("Convergence of SFC Deployment Cost")
plt.savefig("sfc_deployment_convergence.png")
plt.close()

# --------------------------
# 5. Heuristic Function Visualization (GNN outputs)
# --------------------------
heuristic_outputs = []

for i in range(num_samples):
    sfc_sample = sfc_requests[i:i+1]
    h_out = gnn(sfc_sample)
    heuristic_outputs.append(h_out.mean().item())

plt.plot(heuristic_outputs)
plt.xlabel("SFC sample index")
plt.ylabel("Heuristic function value")
plt.title("Heuristic Function Visualization")
plt.savefig("heuristic_visualization.png")
plt.close()

# --------------------------
# 6. Cost Performance Comparison
# --------------------------
plt.plot(vnf_embedding_costs, label="VNF embedding cost")
plt.plot(sfc_deployment_costs, label="SFC deployment cost")
plt.xlabel("SFC sample index")
plt.ylabel("Cost")
plt.title("Cost Performance Comparison")
plt.legend()
plt.savefig("cost_comparison.png")
plt.close()

# --------------------------
# 7. Success Rate and Delay Performance
# --------------------------
success_rate = []
delay_performance = []

for i in range(num_samples):
    sfc_sample = sfc_requests[i:i+1]
    memory = encoder(sfc_sample)
    probs = decoder(sfc_sample, memory)
    # Dummy example: success if sum > threshold
    success = 1 if torch.sum(probs) > 0 else 0
    delay = 1 / (torch.sum(probs) + 1e-6)      # inverse as dummy delay
    success_rate.append(success)
    delay_performance.append(delay.item())

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(success_rate)
plt.xlabel("SFC sample index")
plt.ylabel("Success rate")
plt.title("Success Rate")

plt.subplot(1,2,2)
plt.plot(delay_performance)
plt.xlabel("SFC sample index")
plt.ylabel("Delay")
plt.title("Delay Performance")
plt.tight_layout()
plt.savefig("success_delay_performance.png")
plt.close()

print("All evaluation metrics computed and plots saved.")
