# # # # import torch
# # # # import torch.optim as optim
# # # # # from code.src.models.gnn_heuristicc import GNNHeuristic
# # # # from models.gnn_hueristic import GNNHeuristic

# # # # from env.sfc_env import SFCEnv

# # # # gnn = GNNHeuristic()
# # # # env = SFCEnv()
# # # # optimizer = optim.Adam(gnn.parameters(), lr=1e-4)

# # # # for epoch in range(80):

# # # #     sfc, nodes, edges = env.reset()

# # # #     nodes = torch.tensor(nodes, dtype=torch.float32)
# # # #     edges = torch.tensor(edges, dtype=torch.float32)

# # # #     heuristic = gnn(nodes[:edges.shape[0]], edges)
# # # #     reward = -torch.mean(heuristic)

# # # #     loss = -reward

# # # #     optimizer.zero_grad()
# # # #     loss.backward()
# # # #     optimizer.step()

# # # #     print(f"GNN Epoch {epoch} | Loss: {loss.item():.4f}")

# # # # torch.save(gnn.state_dict(), "gnn_heuristic.pth")

# # # import argparse
# # # import torch
# # # import torch.optim as optim
# # # from models.gnn_hueristic import GNNHeuristic
# # # from env.sfc_env import SFCEnv

# # # # ------------------ Argument Parser ------------------
# # # parser = argparse.ArgumentParser()
# # # parser.add_argument('--data', type=str, required=True, help='Path to topology GML file')
# # # parser.add_argument('--save', type=str, required=True, help='Path to save model checkpoint')
# # # parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
# # # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# # # args = parser.parse_args()

# # # # ------------------ Initialize Environment ------------------
# # # env = SFCEnv(args.data)

# # # # ------------------ Initialize Model and Optimizer ------------------
# # # gnn = GNNHeuristic()
# # # optimizer = optim.Adam(gnn.parameters(), lr=args.lr)

# # # # ------------------ Training Loop ------------------
# # # for epoch in range(args.epochs):
# # #     # Reset environment and get sample SFC data
# # #     sfc, nodes, edges = env.reset()

# # #     # Convert nodes and edges to tensors
# # #     nodes = torch.tensor(nodes, dtype=torch.float32)
# # #     edges = torch.tensor(edges, dtype=torch.float32)

# # #     # Forward pass: compute heuristic
# # #     heuristic = gnn(nodes[:edges.shape[0]], edges)

# # #     # Compute reward (negative heuristic as a cost)
# # #     reward = -torch.mean(heuristic)
# # #     loss = -reward  # maximize reward

# # #     # Backpropagation
# # #     optimizer.zero_grad()
# # #     loss.backward()
# # #     optimizer.step()

# # #     print(f"GNN Epoch {epoch+1} | Loss: {loss.item():.4f}")

# # # # ------------------ Save Model ------------------
# # # torch.save(gnn.state_dict(), args.save)
# # # print(f"GNN Heuristic model saved at {args.save}")
# # import argparse
# # import torch
# # import torch.optim as optim
# # from models.gnn_hueristic import GNNHeuristic
# # from env.sfc_env import SFCEnv

# # # ----------------- Argument Parser -----------------
# # parser = argparse.ArgumentParser(description="Train GNN Heuristic for Transformer-ACO")
# # parser.add_argument('--data', type=str, required=True, help='Path to topology file (e.g., walker66.gml)')
# # parser.add_argument('--save', type=str, required=True, help='Path to save trained model')
# # parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
# # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# # args = parser.parse_args()

# # # ----------------- Initialize Model & Environment -----------------
# # gnn = GNNHeuristic()
# # env = SFCEnv(args.data)  # Pass topology file to environment
# # optimizer = optim.Adam(gnn.parameters(), lr=args.lr)

# # # ----------------- Training Loop -----------------
# # for epoch in range(args.epochs):
# #     # Reset environment and get network topology
# #     reset_data = env.reset()
    
# #     # Adjust according to actual reset() return type
# #     if isinstance(reset_data, dict):
# #         nodes = torch.tensor(reset_data['nodes'], dtype=torch.float32)
# #         edges = torch.tensor(reset_data['edges'], dtype=torch.float32)
# #         sfc = reset_data.get('sfc', None)  # Optional
# #     else:
# #         # If reset() returns tuple like (nodes, edges)
# #         nodes, edges = reset_data
# #         nodes = torch.tensor(nodes, dtype=torch.float32)
# #         edges = torch.tensor(edges, dtype=torch.float32)
# #         sfc = None

# #     # Forward pass through GNN
# #     heuristic = gnn(nodes[:edges.shape[0]], edges)
# #     reward = -torch.mean(heuristic)
    
# #     # Loss & optimization
# #     loss = -reward
# #     optimizer.zero_grad()
# #     loss.backward()
# #     optimizer.step()

# #     print(f"GNN Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")

# # # ----------------- Save Model -----------------
# # torch.save(gnn.state_dict(), args.save)
# # print(f"Model saved at {args.save}")
# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import networkx as nx
# from models.gnn_hueristic import GNNHeuristic  # Ensure correct filename
# from env.sfc_env import SFCEnv                  # Ensure this exists

# # ----------------- ARGUMENTS -----------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--data", type=str, required=True, help="Path to topology file (.gml)")
# parser.add_argument("--save", type=str, default="checkpoints/gnn_heuristic.pth", help="Path to save trained model")
# parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
# parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
# args = parser.parse_args()

# # ----------------- CHECKPOINT FOLDER -----------------
# os.makedirs(os.path.dirname(args.save), exist_ok=True)

# # ----------------- ENVIRONMENT -----------------
# env = SFCEnv(args.data)

# # ----------------- MODEL -----------------
# gnn = GNNHeuristic()
# optimizer = optim.Adam(gnn.parameters(), lr=args.lr)
# loss_fn = nn.MSELoss()  # Example loss (adjust if needed)

# # ----------------- TRAINING LOOP -----------------
# for epoch in range(args.epochs):
#     reset_data = env.reset()  # Should return nodes, edges (modify env.reset if needed)
    
#     # Unpack properly
#     if len(reset_data) == 2:
#         nodes, edges = reset_data
#     else:
#         # fallback if env.reset returns only graph
#         nodes, edges = reset_data, reset_data
    
#     nodes = torch.tensor(nodes, dtype=torch.float32)
#     edges = torch.tensor(edges, dtype=torch.float32)
    
#     # Forward pass through GNN
#     heuristic = gnn(nodes[:edges.shape[0]], edges)
    
#     # Example reward / loss (negative heuristic mean)
#     reward = -torch.mean(heuristic)
#     loss = -reward
    
#     # Backprop
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print(f"GNN Epoch {epoch} | Loss: {loss.item():.4f}")

# # ----------------- SAVE MODEL -----------------
# torch.save(gnn.state_dict(), args.save)
# print(f"GNN heuristic model saved to {args.save}")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import os

# ---------- GNN Heuristic Model ----------
class GNNHeuristic(nn.Module):
    def __init__(self, node_in=3, edge_in=2, hidden=128):
        super(GNNHeuristic, self).__init__()
        self.node_fc = nn.Linear(node_in, hidden)
        self.edge_fc = nn.Linear(edge_in, hidden)
        self.out_fc = nn.Linear(hidden, 1)

    def forward(self, node_features, edge_features):
        # node_features: (num_nodes, node_in)
        # edge_features: (num_edges, edge_in)
        node_h = torch.relu(self.node_fc(node_features))
        edge_h = torch.relu(self.edge_fc(edge_features))
        # simple aggregation: sum node and edge features
        combined = node_h[:edge_h.shape[0]] + edge_h
        out = self.out_fc(combined)
        return out.squeeze(-1)  # shape: (num_edges,)


# ---------- SFC Environment ----------
class SFCEnv:
    def __init__(self, topology_file):
        self.graph = nx.read_gml(topology_file)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        # Create dummy node and edge features
        self.node_features = torch.randn(self.num_nodes, 3)  # 3 features per node
        self.edge_features = torch.randn(self.num_edges, 2)  # 2 features per edge

    def reset(self):
        # For each episode, return node features and edge features
        return self.node_features, self.edge_features


# ---------- Main Training ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/topology/walker66.gml', help='Topology GML file')
    parser.add_argument('--save', type=str, default='checkpoints/gnn_heuristic.pth', help='Path to save checkpoint')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # Create checkpoints directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # Initialize model, environment, optimizer
    gnn = GNNHeuristic()
    env = SFCEnv(args.data)
    optimizer = optim.Adam(gnn.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        nodes, edges = env.reset()  # tensors
        # Forward pass
        heuristic = gnn(nodes, edges)
        # Reward: negative mean heuristic (to minimize)
        reward = -torch.mean(heuristic)
        loss = -reward  # maximize reward

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"GNN Epoch {epoch} | Loss: {loss.item():.4f}")

    # Save checkpoint
    torch.save(gnn.state_dict(), args.save)
    print(f"Checkpoint saved at {args.save}")


if __name__ == "__main__":
    main()
