# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import networkx as nx
# # import matplotlib.pyplot as plt
# # from models.encoder import SFCEncoder
# # from models.decoder import SFCDecoder
# # from models.gnn_hueristic import GNNHeuristic  # adjust if your class name differs

# # # -----------------------------
# # # 1. Load models
# # # -----------------------------
# # encoder = SFCEncoder()
# # encoder.load_state_dict(torch.load("checkpoints/encoder.pth"))
# # encoder.eval()

# # decoder = SFCDecoder()
# # decoder.load_state_dict(torch.load("checkpoints/decoder.pth"))
# # decoder.eval()

# # gnn = GNNHeuristic()
# # gnn.load_state_dict(torch.load("checkpoints/gnn_heuristic.pth"))
# # gnn.eval()

# # # -----------------------------
# # # 2. Load data
# # # -----------------------------
# # sfc_requests = np.load("data/sfc_requests.npy")  # (num_requests, seq_len, features)
# # G = nx.read_gml("data/topology/walker66.gml")

# # # -----------------------------
# # # 3. Placeholder cost/delay functions
# # # -----------------------------
# # def compute_vnf_cost(actions):
# #     return actions.sum().item()

# # def compute_total_cost(actions, path_cost):
# #     return compute_vnf_cost(actions) + path_cost.sum().item()

# # def compute_delay(sfc):
# #     return np.random.rand()  # replace with your actual delay calculation

# # # -----------------------------
# # # 4. VNF embedding cost
# # # -----------------------------
# # vnf_embedding_costs = []
# # for sfc in sfc_requests:
# #     sfc_tensor = torch.tensor(sfc, dtype=torch.float).unsqueeze(0)  # batch_size=1
# #     with torch.no_grad():
# #         embed = encoder(sfc_tensor)
# #         actions = decoder(embed, embed)
# #     cost = compute_vnf_cost(actions)
# #     vnf_embedding_costs.append(cost)
# # vnf_embedding_costs = np.array(vnf_embedding_costs)
# # np.save("results/vnf_embedding_costs.npy", vnf_embedding_costs)

# # # -----------------------------
# # # 5. SFC deployment cost
# # # -----------------------------
# # sfc_deployment_costs = []
# # for sfc in sfc_requests:
# #     sfc_tensor = torch.tensor(sfc, dtype=torch.float).unsqueeze(0)
# #     with torch.no_grad():
# #         embed = encoder(sfc_tensor)
# #         actions = decoder(embed, embed)
# #         node_features = torch.randn(len(G.nodes), 6)  # example node features
# #         edges = torch.randn(len(G.edges), 2)          # example edges
# #         path_cost = gnn(node_features, edges)
# #     total_cost = compute_total_cost(actions, path_cost)
# #     sfc_deployment_costs.append(total_cost)
# # sfc_deployment_costs = np.array(sfc_deployment_costs)
# # np.save("results/sfc_deployment_costs.npy", sfc_deployment_costs)

# # # -----------------------------
# # # 6. Heuristic function visualization
# # # -----------------------------
# # node_features = torch.randn(len(G.nodes), 6)
# # edges = torch.randn(len(G.edges), 2)
# # with torch.no_grad():
# #     heuristic_scores = gnn(node_features, edges).numpy()

# # plt.figure(figsize=(8,6))
# # nx.draw(G, with_labels=True, node_color=heuristic_scores, cmap=plt.cm.viridis, node_size=400)
# # plt.title("Heuristic Function Visualization")
# # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Heuristic Score")
# # plt.savefig("results/heuristic_visualization.png")
# # plt.show()

# # # -----------------------------
# # # 7. Cost performance comparison
# # # -----------------------------
# # methods = ['Random', 'Greedy', 'Our Method']
# # random_cost = np.random.rand() * 100
# # greedy_cost = np.random.rand() * 100
# # our_method_cost = sfc_deployment_costs.mean()

# # costs = [random_cost, greedy_cost, our_method_cost]
# # np.save("results/cost_comparison.npy", np.array(costs))

# # # -----------------------------
# # # 8. Success rate & delay
# # # -----------------------------
# # success_threshold = 50
# # success_rate = np.sum(sfc_deployment_costs < success_threshold) / len(sfc_requests)

# # delays = [compute_delay(sfc) for sfc in sfc_requests]
# # avg_delay = np.mean(delays)

# # np.save("results/success_rate.npy", np.array([success_rate]))
# # np.save("results/avg_delay.npy", np.array([avg_delay]))

# # print("Evaluation completed. All matrices saved in 'results/' folder.")
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# from models.encoder import SFCEncoder
# from models.decoder import SFCDecoder

# # Modified GNNHeuristic to match your checkpoint
# class GNNHeuristic(nn.Module):
#     def __init__(self):
#         super(GNNHeuristic, self).__init__()
#         self.node_fc = nn.Linear(3, 128)
#         self.edge_fc = nn.Linear(2, 128)
#         self.out_fc = nn.Linear(128, 1)

#     def forward(self, node_features, edge_features):
#         node_h = torch.relu(self.node_fc(node_features))
#         edge_h = torch.relu(self.edge_fc(edge_features))
#         h = node_h + edge_h.mean(dim=0, keepdim=True)
#         out = self.out_fc(h)
#         return out

# # Ensure folders exist
# os.makedirs("plots", exist_ok=True)

# # Load checkpoints
# encoder = SFCEncoder()
# encoder.load_state_dict(torch.load("checkpoints/encoder.pth"))
# encoder.eval()

# decoder = SFCDecoder()
# decoder.load_state_dict(torch.load("checkpoints/decoder.pth"))
# decoder.eval()

# gnn = GNNHeuristic()
# gnn.load_state_dict(torch.load("checkpoints/gnn_heuristic.pth"))
# gnn.eval()

# # Dummy test data for evaluation
# # Replace with actual test SFC requests if available
# num_samples = 50
# seq_len = 10
# node_features_dim = 3
# edge_features_dim = 2

# # For demonstration, random data
# node_features = torch.rand(num_samples, node_features_dim)
# edge_features = torch.rand(num_samples, edge_features_dim)
# sfc_requests = torch.rand(num_samples, seq_len, 8)  # 8 features per request

# # ----- 1. VNF Embedding Cost Convergence -----
# vnf_embedding_costs = []
# for i in range(num_samples):
#     embed = encoder(sfc_requests[i:i+1])
#     cost = torch.norm(embed).item()
#     vnf_embedding_costs.append(cost)

# plt.plot(vnf_embedding_costs, label="VNF Embedding Cost")
# plt.xlabel("Sample")
# plt.ylabel("Cost")
# plt.title("Convergence of VNF Embedding Cost")
# plt.legend()
# plt.savefig("plots/vnf_embedding_convergence.png")
# plt.clf()

# # ----- 2. SFC Deployment Cost Convergence -----
# deployment_costs = []
# for i in range(num_samples):
#     memory = encoder(sfc_requests[i:i+1])
#     probs = decoder(sfc_requests[i:i+1], memory)
#     cost = -torch.sum(probs).item()  # dummy deployment cost
#     deployment_costs.append(cost)

# plt.plot(deployment_costs, label="SFC Deployment Cost", color="orange")
# plt.xlabel("Sample")
# plt.ylabel("Cost")
# plt.title("Convergence of SFC Deployment Cost")
# plt.legend()
# plt.savefig("plots/sfc_deployment_convergence.png")
# plt.clf()

# # ----- 3. Heuristic Function Visualization -----
# heuristic_values = []
# for i in range(num_samples):
#     out = gnn(node_features[i:i+1], edge_features[i:i+1])
#     heuristic_values.append(out.item())

# plt.plot(heuristic_values, label="Heuristic Value", color="green")
# plt.xlabel("Sample")
# plt.ylabel("Heuristic")
# plt.title("Heuristic Function Visualization")
# plt.legend()
# plt.savefig("plots/heuristic_visualization.png")
# plt.clf()

# # ----- 4. Cost Performance Comparison -----
# # Random comparison between GNN heuristic and decoder cost
# decoder_costs = deployment_costs
# gnn_costs = heuristic_values[:len(decoder_costs)]  # same length

# plt.plot(decoder_costs, label="Decoder Deployment Cost")
# plt.plot(gnn_costs, label="GNN Heuristic Cost")
# plt.xlabel("Sample")
# plt.ylabel("Cost")
# plt.title("Cost Performance Comparison")
# plt.legend()
# plt.savefig("plots/cost_comparison.png")
# plt.clf()

# # ----- 5. Success Rate and Delay Performance -----
# # Dummy metrics: success=1 if cost<threshold
# threshold = np.median(deployment_costs)
# success_rate = [1 if c < threshold else 0 for c in deployment_costs]
# delay = np.random.rand(num_samples) * 10  # dummy delay

# plt.plot(success_rate, label="Success Rate", color="blue")
# plt.plot(delay, label="Delay", color="red")
# plt.xlabel("Sample")
# plt.ylabel("Value")
# plt.title("Success Rate and Delay Performance")
# plt.legend()
# plt.savefig("plots/success_delay_performance.png")
# plt.clf()

# print("All evaluation plots saved in 'plots/' directory.")
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
