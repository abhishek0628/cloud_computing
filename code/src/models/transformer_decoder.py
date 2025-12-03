# src/models/transformer_decoder.py
"""
Transformer Decoder for auto-regressive VNF placement along a given ACO path.

Given:
  - encoded VNFs: (seq_len_vnf, d_model)
  - node features along path: (path_len, node_feature_dim)
  
Produces:
  - embedding_list: list mapping VNF index -> selected node index on path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self, 
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 node_feature_dim=3,   # cpu, ram, energy
                 dim_feedforward=128,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.node_feature_dim = node_feature_dim

        # project raw node features -> d_model
        self.node_fc = nn.Linear(node_feature_dim, d_model)

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # scoring head: score = f(decoder_output, node_embeddings)
        self.scorer = nn.Linear(d_model, d_model)

        # initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_fc.weight)
        if self.node_fc.bias is not None:
            nn.init.zeros_(self.node_fc.bias)

    def forward(self, encoded_vnfs, path_node_features):
        """
        encoded_vnfs: (seq_len_vnf, d_model)
        path_node_features: (path_len, node_feature_dim)

        returns:
            embedding_list: list of selected node positions for each VNF
        """

        # Convert to batch form
        encoded_vnfs = encoded_vnfs.unsqueeze(0)   # (1, seq_len, d_model)

        # Convert path node features -> embeddings
        node_emb = self.node_fc(path_node_features)  # (path_len, d_model)
        node_emb = node_emb.unsqueeze(0)             # (1, path_len, d_model)

        embedding_list = []   # store selected node positions
        path_len = node_emb.size(1)
        seq_len = encoded_vnfs.size(1)

        # auto-regressive decoding
        prev_outputs = torch.zeros_like(encoded_vnfs)  # initial zero context

        for k in range(seq_len):
            # use encoded_vnf[k] + previous decoded results
            tgt = prev_outputs[:, :k+1, :]    # (1, k+1, d_model)

            # Prepare memory for decoder
            memory = node_emb                 # (1, path_len, d_model)

            # Decoder forward pass for position k
            dec_out = self.decoder(tgt, memory)  # (1, k+1, d_model)
            last_token = dec_out[:, -1, :]       # (1, d_model)

            # Score each node along path
            scores = torch.matmul(
                self.scorer(last_token),      # (1, d_model)
                node_emb.squeeze(0).t()       # (d_model, path_len)
            )  # -> (1, path_len)

            probs = F.softmax(scores, dim=-1)       # (1, path_len)
            selected_node = torch.argmax(probs, dim=-1).item()

            embedding_list.append(selected_node)

            # inject chosen node embedding into prev_outputs
            prev_outputs[:, k, :] = node_emb[:, selected_node, :]

        return embedding_list
