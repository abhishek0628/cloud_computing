
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# GNN Heuristic Model
# -----------------------------
class GNNHeuristic(nn.Module):
    def __init__(self):
        super(GNNHeuristic, self).__init__()
        # match checkpoint dimensions
        self.node_fc = nn.Linear(3, 128)    # node features = 3
        self.edge_fc = nn.Linear(2, 128)    # edge features = 2
        self.out_fc  = nn.Linear(128, 1)    # output hidden = 128

    def forward(self, node_features, edge_features):
        h_node = torch.relu(self.node_fc(node_features))
        h_edge = torch.relu(self.edge_fc(edge_features))
        h = h_node + h_edge  # simple combination
        out = self.out_fc(h)
        return out

# -----------------------------
# Load checkpoint
# -----------------------------
gnn = GNNHeuristic()
checkpoint = torch.load("checkpoints/gnn_heuristic.pth")
gnn.load_state_dict(checkpoint)
gnn.eval()

# -----------------------------
# Load test data
# -----------------------------
# Replace with your actual node/edge data
num_samples = 100
node_features = torch.randn(num_samples, 3)  # matches checkpoint
edge_features = torch.randn(num_samples, 2)

# -----------------------------
# Run GNN and collect heuristic outputs
# -----------------------------
with torch.no_grad():
    heuristic_outputs = gnn(node_features, edge_features)

# -----------------------------
# Plot heuristic function
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(heuristic_outputs.numpy(), marker='o', linestyle='', alpha=0.7)
plt.title("Heuristic Function Visualization")
plt.xlabel("Sample Index")
plt.ylabel("Heuristic Output")
plt.grid(True)
plt.savefig("./plots/heuristic_visualization.png")
plt.show()
