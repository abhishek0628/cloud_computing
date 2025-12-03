
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
