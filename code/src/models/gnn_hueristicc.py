# src/models/gnn_hueristic.py
"""
GNN-based heuristic generator (η_theta) for ACO.

This module:
  - Encodes graph nodes using a simple GraphSAGE-like GNN
  - Computes edge embeddings
  - Performs "resource-node matching attention" between SFC VNF embeddings & edge embeddings
  - Passes attention results through an MLP to get heuristic score η(i, j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    """Simple GraphSAGE layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, node_feats, adjacency_list):
        """
        node_feats: (N, F)
        adjacency_list: dict {node: [neighbors]}
        """
        N = node_feats.size(0)
        out = torch.zeros_like(node_feats)

        for i in range(N):
            neighs = adjacency_list.get(i, [])
            if len(neighs) == 0:
                neigh_vec = torch.zeros_like(node_feats[i])
            else:
                neigh_vec = torch.mean(node_feats[neighs], dim=0)

            out[i] = self.linear_self(node_feats[i]) + self.linear_neigh(neigh_vec)

        return F.relu(out)


class GNNHeuristic(nn.Module):
    """
    Given:
      - Graph node features (CPU, RAM, Energy)
      - Edge features (delay, bandwidth, duration)
      - Encoded VNFs (from Transformer Encoder)

    Produces:
      - heuristic matrix η(i,j) for each edge in graph
    """

    def __init__(self, node_in_dim=3, edge_in_dim=3, d_model=64, gnn_hidden=64, num_gnn_layers=2):
        super().__init__()

        self.node_fc = nn.Linear(node_in_dim, gnn_hidden)
        self.edge_fc = nn.Linear(edge_in_dim, gnn_hidden)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphSAGE(gnn_hidden, gnn_hidden) for _ in range(num_gnn_layers)
        ])

        # Attention: combine edge embeddings + VNF embeddings
        self.attention_fc = nn.Linear(gnn_hidden + d_model, gnn_hidden)

        # MLP to produce heuristic score η(i,j)
        self.scorer = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_fc.weight)
        nn.init.xavier_uniform_(self.edge_fc.weight)
        for layer in self.gnn_layers:
            nn.init.xavier_uniform_(layer.linear_self.weight)
            nn.init.xavier_uniform_(layer.linear_neigh.weight)

    def forward(self, G, encoded_vnfs):
        """
        G: NetworkX graph
        encoded_vnfs: (num_vnfs, d_model)

        Returns:
            heuristic_matrix: dict {(i,j): score}
        """

        # 1. Prepare node features
        node_list = list(G.nodes())
        N = len(node_list)

        node_feats = []
        for n in node_list:
            cpu = float(G.nodes[n]['cpu'])
            ram = float(G.nodes[n]['ram'])
            energy = float(G.nodes[n]['energy'])
            node_feats.append([cpu, ram, energy])

        node_feats = torch.tensor(node_feats, dtype=torch.float32)  # (N,3)

        # Project to hidden dim
        node_emb = self.node_fc(node_feats)  # (N, hidden)

        # 2. Build adjacency list
        adjacency_list = {}
        for n in node_list:
            adjacency_list[n] = list(G.neighbors(n))

        # 3. GNN propagation
        for gnn in self.gnn_layers:
            node_emb = gnn(node_emb, adjacency_list)

        # 4. Edge embeddings
        edge_embeddings = {}
        for (i, j) in G.edges():
            edge_features = [
                float(G[i][j]['delay']),
                float(G[i][j]['bandwidth']),
                float(G[i][j]['duration'])
            ]
            ef = torch.tensor(edge_features, dtype=torch.float32)
            edge_embeddings[(i, j)] = self.edge_fc(ef)  # (hidden)

        # 5. Resource-node matching attention:
        # For each edge, match with all VNF embeddings → mean aggregated attention
        vnf_mean = torch.mean(encoded_vnfs, dim=0)  # (d_model)

        heuristic_scores = {}

        for (i, j), e_emb in edge_embeddings.items():
            combined = torch.cat([e_emb, vnf_mean], dim=0)  # (hidden + d_model)
            h = self.attention_fc(combined)                 # (hidden)
            h = torch.relu(h)
            score = self.scorer(h)                         # (1)
            heuristic_scores[(i, j)] = score.item()

        return heuristic_scores
