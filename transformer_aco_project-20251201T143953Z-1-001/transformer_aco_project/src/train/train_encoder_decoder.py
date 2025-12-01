import sys
import os

# -----------------------------------------------------------
# FIX 1: Add correct src path
# -----------------------------------------------------------
BASE = "/content/drive/MyDrive/transformer_aco_project/src"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# -----------------------------------------------------------
# FIX 2: Ensure __init__.py exists so Python sees modules
# -----------------------------------------------------------
os.makedirs(BASE, exist_ok=True)
open(os.path.join(BASE, "__init__.py"), 'a').close()
open(os.path.join(BASE, "env", "__init__.py"), 'a').close()
open(os.path.join(BASE, "models", "__init__.py"), 'a').close()
open(os.path.join(BASE, "train", "__init__.py"), 'a').close()


# -----------------------------------------------------------
# IMPORTS (will now work)
# -----------------------------------------------------------
from env.topology import generate_topology
from env.sfc_request import generate_sfc
from env.simulator import compute_resource_variance

from models.transformer_encoder import TransformerEncoderModel
from models.transformer_decoder import TransformerDecoder

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import networkx as nx


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------
def extract_path_node_features(G, path):
    feats = []
    for n in path:
        feats.append([
            float(G.nodes[n]['cpu']),
            float(G.nodes[n]['ram']),
            float(G.nodes[n]['energy']),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def reinforce_loss(resource_variance, log_probs, gamma=1.0):
    total_log_prob = torch.stack(log_probs).sum()
    return gamma * resource_variance * total_log_prob


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------
def train_reinforce(
    num_epochs=3,
    sfc_per_epoch=10,
    samples_per_sfc=2,
    lr=0.0005,
    topology_nodes=20,
    save_path="/content/drive/MyDrive/transformer_aco_project/experiments/encoder_decoder.pth"
):

    print("Generating topology...")
    G = generate_topology(topology_nodes)

    encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
    decoder = TransformerDecoder(d_model=64)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    print("Starting REINFORCE training...\n")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        for _ in tqdm(range(sfc_per_epoch), desc=f"Epoch {epoch}"):

            sfc = generate_sfc(k=5)
            vnf_tensor = torch.tensor(
                [[v['cpu'], v['ram'], v['bw'], v['energy'], v['duration']] for v in sfc],
                dtype=torch.float32
            )

            encoded = encoder(vnf_tensor)

            nodes = list(G.nodes())
            src, dst = random.sample(nodes, 2)

            try:
                simple_path = nx.shortest_path(G, src, dst)
            except:
                continue

            path_features = extract_path_node_features(G, simple_path)

            for _ in range(samples_per_sfc):
                log_probs = []
                placement = []

                for k in range(len(encoded)):
                    dec_out = decoder.decoder(
                        encoded.unsqueeze(0)[:, :k+1, :],
                        decoder.node_fc(path_features).unsqueeze(0)
                    )[:, -1, :]

                    scores = dec_out @ decoder.node_fc(path_features).t()
                    probs = nn.Softmax(dim=-1)(scores)

                    dist = torch.distributions.Categorical(probs.squeeze(0))
                    action = dist.sample()

                    placement.append(action.item())
                    log_probs.append(dist.log_prob(action))

                chosen_nodes = [simple_path[i] for i in placement]
                variance = compute_resource_variance(G, chosen_nodes)

                loss = reinforce_loss(variance, log_probs)
                epoch_loss += loss.item()

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch} | Avg Loss = {epoch_loss / (sfc_per_epoch * samples_per_sfc):.4f}")

        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        }, save_path)

    print(f"\nTraining complete. Saved to {save_path}")


if __name__ == "__main__":
    train_reinforce()
