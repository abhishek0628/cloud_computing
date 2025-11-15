import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallGraphTransformer(nn.Module):
    def __init__(self, node_feat_dim=6, d_model=64, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.node_score = nn.Linear(d_model, 1)
        # edge scorer: take concat of node embeddings (i||j)
        self.edge_score = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, node_feats, adj_mask=None):
        """
        node_feats: (N, node_feat_dim) float tensor
        adj_mask unused here but kept for API parity
        returns: node_logits (N,), edge_logits (N,N)
        """
        x = self.node_proj(node_feats)            # (N, d)
        # transformer expects (S, B, d). use B=1
        x = x.unsqueeze(1)                       # (N,1,d)
        x = self.transformer(x)                  # (N,1,d)
        x = x.squeeze(1)                         # (N,d)
        node_logits = self.node_score(x).squeeze(-1)  # (N,)
        # build edge logits via outer concat (inefficient but small N)
        N, d = x.shape
        xi = x.unsqueeze(1).expand(N, N, d)
        xj = x.unsqueeze(0).expand(N, N, d)
        edges = torch.cat([xi, xj], dim=-1)      # (N,N,2d)
        edge_logits = self.edge_score(edges).squeeze(-1)  # (N,N)
        # some self-edges may be meaningless; we will mask later in ACO
        return node_logits, edge_logits
