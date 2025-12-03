import torch
import torch.nn as nn

class GNNHeuristic(nn.Module):
    def __init__(self, node_dim=6, edge_dim=4, hidden_dim=128):
        super().__init__()

        self.node_fc = nn.Linear(node_dim, hidden_dim)
        self.edge_fc = nn.Linear(edge_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, node_features, edge_features):
        node_h = torch.relu(self.node_fc(node_features))
        edge_h = torch.relu(self.edge_fc(edge_features))

        combined = torch.cat([node_h, edge_h], dim=-1)
        heuristic = self.out_fc(combined)

        return torch.sigmoid(heuristic)
