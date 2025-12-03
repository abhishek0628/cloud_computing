# # src/eval/evaluate_encoder_decoder.py
# import sys
# import os
# import torch
# import random
# import networkx as nx
# import numpy as np

# BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if BASE not in sys.path:
#     sys.path.insert(0, BASE)

# from env.topology import generate_topology
# from env.sfc_request import generate_sfc
# from env.simulator import compute_resource_variance
# from models.transformer_encoder import TransformerEncoderModel
# from models.transformer_decoder import TransformerDecoder


# # MODEL_PATH = os.path.join(BASE, "experiments", "encoder_decoder.pth")
# MODEL_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "encoder_decoder.pth")
# )



# def extract_path_node_features(G, path):
#     feats = []
#     for n in path:
#         feats.append([
#             float(G.nodes[n]['cpu']),
#             float(G.nodes[n]['ram']),
#             float(G.nodes[n]['energy']),
#         ])
#     return torch.tensor(feats, dtype=torch.float32)


# def evaluate(num_tests=30):
#     print("Loading topology...")
#     G = generate_topology(20)

#     print("Loading model...")
#     ckpt = torch.load(MODEL_PATH, map_location="cpu")

#     encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
#     decoder = TransformerDecoder(d_model=64)

#     encoder.load_state_dict(ckpt["encoder"])
#     decoder.load_state_dict(ckpt["decoder"])

#     encoder.eval()
#     decoder.eval()

#     rl_scores = []
#     rnd_scores = []

#     for _ in range(num_tests):
#         sfc = generate_sfc(k=5)

#         vnf_tensor = torch.tensor(
#             [[v['cpu'], v['ram'], v['bw'], v['energy'], v['duration']] for v in sfc],
#             dtype=torch.float32
#         )

#         encoded = encoder(vnf_tensor)

#         src, dst = random.sample(list(G.nodes()), 2)

#         try:
#             path = nx.shortest_path(G, src, dst)
#         except:
#             continue

#         path_features = extract_path_node_features(G, path)

#         placement = []

#         for k in range(len(encoded)):
#             dec_out = decoder.decoder(
#                 encoded.unsqueeze(0)[:, :k+1, :],
#                 decoder.node_fc(path_features).unsqueeze(0)
#             )[:, -1, :]

#             scores = dec_out @ decoder.node_fc(path_features).t()
#             probs = torch.softmax(scores, dim=-1)

#             choice = torch.argmax(probs).item()
#             placement.append(choice)

#         # ✅ FIXED HERE
#         placement = [min(p, len(path) - 1) for p in placement]
#         chosen_nodes = [path[p] for p in placement]
#         rl_var = compute_resource_variance(G, chosen_nodes)
#         rl_scores.append(rl_var)

#         k = min(len(path), len(placement))
#         rand_nodes = random.sample(path, k=k)
#         rnd_var = compute_resource_variance(G, rand_nodes)
#         rnd_scores.append(rnd_var)

#     return np.mean(rl_scores), np.mean(rnd_scores)


# if __name__ == "__main__":
#     rl, rnd = evaluate()
#     print("\nRL Avg Variance:", rl)
#     print("Random Avg Variance:", rnd)

import sys
import os
import torch
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from env.topology import generate_topology
from env.sfc_request import generate_sfc
from env.simulator import compute_resource_variance
from models.transformer_encoder import TransformerEncoderModel
from models.transformer_decoder import TransformerDecoder

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "encoder_decoder.pth")
)

def extract_path_node_features(G, path):
    feats = []
    for n in path:
        feats.append([
            float(G.nodes[n]['cpu']),
            float(G.nodes[n]['ram']),
            float(G.nodes[n]['energy']),
        ])
    return torch.tensor(feats, dtype=torch.float32)

def plot_topology_with_path(G, path):
    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=300)

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            width=3
        )

    plt.title("Network Topology with Selected Path")
    plt.savefig("plots/network_path.png")
    plt.show()

def evaluate(num_tests=30):
    print("Loading topology...")
    G = generate_topology(20)

    print("Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
    decoder = TransformerDecoder(d_model=64)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    rl_scores = []
    rnd_scores = []

    for it in range(num_tests):
        sfc = generate_sfc(k=5)

        vnf_tensor = torch.tensor(
            [[v['cpu'], v['ram'], v['bw'], v['energy'], v['duration']] for v in sfc],
            dtype=torch.float32
        )

        encoded = encoder(vnf_tensor)

        src, dst = random.sample(list(G.nodes()), 2)

        try:
            path = nx.shortest_path(G, src, dst)
        except:
            continue

        if it == 0:
            plot_topology_with_path(G, path)

        path_features = extract_path_node_features(G, path)

        placement = []

        for k in range(len(encoded)):
            dec_out = decoder.decoder(
                encoded.unsqueeze(0)[:, :k+1, :],
                decoder.node_fc(path_features).unsqueeze(0)
            )[:, -1, :]

            scores = dec_out @ decoder.node_fc(path_features).t()
            probs = torch.softmax(scores, dim=-1)

            choice = torch.argmax(probs).item()
            placement.append(choice)

        placement = [min(p, len(path) - 1) for p in placement]
        chosen_nodes = [path[p] for p in placement]
        rl_var = compute_resource_variance(G, chosen_nodes)
        rl_scores.append(rl_var)

        k = min(len(path), len(placement))
        rand_nodes = random.sample(path, k=k)
        rnd_var = compute_resource_variance(G, rand_nodes)
        rnd_scores.append(rnd_var)

    # ------------------ BOXPLOT ------------------
    plt.figure()
    plt.boxplot([rl_scores, rnd_scores], labels=["RL", "Random"])
    plt.ylabel("Resource Variance")
    plt.title("RL vs Random VNF Placement")
    plt.grid(True)
    plt.savefig("rl_vs_random.png")
    plt.show()

    return np.mean(rl_scores), np.mean(rnd_scores)

if __name__ == "__main__":
    rl, rnd = evaluate()
    # print("\nRL Avg Variance:", rl)
    # print("Random Avg Variance:", rnd)
    # print("Plots saved: training_loss.png, rl_vs_random.png, network_path.png")
