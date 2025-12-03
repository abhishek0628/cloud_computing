

# # # src/train/train_encoder_decoder.py
# # import os
# # import sys

# # # -----------------------------------------------------------
# # # HARD PATH FIX (NO ENV VARS REQUIRED)
# # # -----------------------------------------------------------
# # THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# # SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
# # if SRC_DIR not in sys.path:
# #     sys.path.insert(0, SRC_DIR)

# # # -----------------------------------------------------------
# # # IMPORTS
# # # -----------------------------------------------------------
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import random
# # import networkx as nx
# # from tqdm import tqdm

# # from env.topology import generate_topology
# # from env.sfc_request import generate_sfc
# # from env.simulator import compute_resource_variance
# # from models.transformer_encoder import TransformerEncoderModel
# # from models.transformer_decoder import TransformerDecoder

# # # -----------------------------------------------------------
# # # PATH NODE FEATURE EXTRACTION
# # # -----------------------------------------------------------
# # def extract_path_node_features(G, path):
# #     feats = []
# #     for n in path:
# #         feats.append([
# #             float(G.nodes[n]['cpu']),
# #             float(G.nodes[n]['ram']),
# #             float(G.nodes[n]['energy'])
# #         ])
# #     return torch.tensor(feats, dtype=torch.float32)

# # # -----------------------------------------------------------
# # # REINFORCE LOSS
# # # -----------------------------------------------------------
# # def reinforce_loss(resource_variance, log_probs, gamma=1.0):
# #     total_log_prob = torch.stack(log_probs).sum()
# #     return gamma * resource_variance * total_log_prob

# # # -----------------------------------------------------------
# # # TRAINING LOOP
# # # -----------------------------------------------------------
# # def train_reinforce(
# #     num_epochs=3,
# #     sfc_per_epoch=10,
# #     samples_per_sfc=2,
# #     lr=0.0005,
# #     topology_nodes=20,
# # ):

# #     PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
# #     SAVE_PATH = os.path.join(PROJECT_ROOT, "experiments", "encoder_decoder.pth")

# #     print("Generating topology...")
# #     G = generate_topology(topology_nodes)

# #     encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
# #     decoder = TransformerDecoder(d_model=64)

# #     params = list(encoder.parameters()) + list(decoder.parameters())
# #     optimizer = optim.Adam(params, lr=lr)

# #     print("Starting REINFORCE training...\n")

# #     for epoch in range(1, num_epochs + 1):
# #         epoch_loss = 0.0

# #         for _ in tqdm(range(sfc_per_epoch), desc=f"Epoch {epoch}"):

# #             sfc = generate_sfc(k=5)

# #             vnf_tensor = torch.tensor(
# #                 [[v['cpu'], v['ram'], v['bw'], v['energy'], v['duration']] for v in sfc],
# #                 dtype=torch.float32
# #             )

# #             encoded = encoder(vnf_tensor)

# #             nodes = list(G.nodes())
# #             src, dst = random.sample(nodes, 2)

# #             try:
# #                 simple_path = nx.shortest_path(G, src, dst)
# #             except:
# #                 continue

# #             path_features = extract_path_node_features(G, simple_path)

# #             optimizer.zero_grad()
# #             total_loss = 0.0

# #             for _ in range(samples_per_sfc):
# #                 log_probs = []
# #                 placement = []

# #                 for k in range(len(encoded)):
# #                     dec_out = decoder.decoder(
# #                         encoded.unsqueeze(0)[:, :k+1, :],
# #                         decoder.node_fc(path_features).unsqueeze(0)
# #                     )[:, -1, :]

# #                     scores = dec_out @ decoder.node_fc(path_features).t()
# #                     probs = nn.Softmax(dim=-1)(scores)

# #                     dist = torch.distributions.Categorical(probs.squeeze(0))
# #                     action = dist.sample()

# #                     placement.append(action.item())
# #                     log_probs.append(dist.log_prob(action))

# #                 chosen_nodes = [simple_path[i] for i in placement]
# #                 variance = compute_resource_variance(G, chosen_nodes)

# #                 loss = reinforce_loss(variance, log_probs)
# #                 total_loss += loss

# #             epoch_loss += total_loss.item()
# #             total_loss.backward()
# #             optimizer.step()

# #         print(f"Epoch {epoch} | Avg Loss = {epoch_loss / (sfc_per_epoch * samples_per_sfc):.4f}")

# #         os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
# #         torch.save({
# #             "encoder": encoder.state_dict(),
# #             "decoder": decoder.state_dict(),
# #         }, SAVE_PATH)

# #     print(f"\nTraining complete. Saved to {SAVE_PATH}")

# # # -----------------------------------------------------------
# # # MAIN
# # # -----------------------------------------------------------
# # if __name__ == "__main__":
# #     train_reinforce()

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import networkx as nx
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# # -----------------------------------------------------------
# # HARD PATH FIX
# # -----------------------------------------------------------
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)

# # -----------------------------------------------------------
# # IMPORTS
# # -----------------------------------------------------------
# from env.topology import generate_topology
# from env.sfc_request import generate_sfc
# from env.simulator import compute_resource_variance
# from models.transformer_encoder import TransformerEncoderModel
# from models.transformer_decoder import TransformerDecoder

# # -----------------------------------------------------------
# # PATH NODE FEATURE EXTRACTION
# # -----------------------------------------------------------
# def extract_path_node_features(G, path):
#     feats = []
#     for n in path:
#         feats.append([
#             float(G.nodes[n]['cpu']),
#             float(G.nodes[n]['ram']),
#             float(G.nodes[n]['energy'])
#         ])
#     return torch.tensor(feats, dtype=torch.float32)

# # -----------------------------------------------------------
# # REINFORCE LOSS
# # -----------------------------------------------------------
# def reinforce_loss(resource_variance, log_probs, gamma=1.0):
#     total_log_prob = torch.stack(log_probs).sum()
#     return gamma * resource_variance * total_log_prob

# # -----------------------------------------------------------
# # TRAINING LOOP
# # -----------------------------------------------------------
# def train_reinforce(
#     num_epochs=3,
#     sfc_per_epoch=10,
#     samples_per_sfc=2,
#     lr=0.0005,
#     topology_nodes=20,
# ):

#     PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
#     SAVE_PATH = os.path.join(PROJECT_ROOT, "experiments", "encoder_decoder.pth")

#     print("Generating topology...")
#     G = generate_topology(topology_nodes)

#     encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
#     decoder = TransformerDecoder(d_model=64)

#     params = list(encoder.parameters()) + list(decoder.parameters())
#     optimizer = optim.Adam(params, lr=lr)

#     print("Starting REINFORCE training...\n")

#     all_epoch_losses = []

#     for epoch in range(1, num_epochs + 1):
#         epoch_loss = 0.0

#         for _ in tqdm(range(sfc_per_epoch), desc=f"Epoch {epoch}"):

#             sfc = generate_sfc(k=5)

#             vnf_tensor = torch.tensor(
#                 [[v['cpu'], v['ram'], v['bw'], v['energy'], v['duration']] for v in sfc],
#                 dtype=torch.float32
#             )

#             encoded = encoder(vnf_tensor)

#             nodes = list(G.nodes())
#             src, dst = random.sample(nodes, 2)

#             try:
#                 simple_path = nx.shortest_path(G, src, dst)
#             except:
#                 continue

#             path_features = extract_path_node_features(G, simple_path)

#             optimizer.zero_grad()
#             total_loss = 0.0

#             for _ in range(samples_per_sfc):
#                 log_probs = []
#                 placement = []

#                 for k in range(len(encoded)):
#                     dec_out = decoder.decoder(
#                         encoded.unsqueeze(0)[:, :k+1, :],
#                         decoder.node_fc(path_features).unsqueeze(0)
#                     )[:, -1, :]

#                     scores = dec_out @ decoder.node_fc(path_features).t()
#                     probs = nn.Softmax(dim=-1)(scores)

#                     dist = torch.distributions.Categorical(probs.squeeze(0))
#                     action = dist.sample()

#                     placement.append(action.item())
#                     log_probs.append(dist.log_prob(action))

#                 chosen_nodes = [simple_path[i] for i in placement]
#                 variance = compute_resource_variance(G, chosen_nodes)

#                 loss = reinforce_loss(variance, log_probs)
#                 total_loss += loss

#             epoch_loss += total_loss.item()
#             total_loss.backward()
#             optimizer.step()

#         avg_loss = epoch_loss / (sfc_per_epoch * samples_per_sfc)
#         all_epoch_losses.append(avg_loss)

#         print(f"Epoch {epoch} | Avg Loss = {avg_loss:.4f}")

#         os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
#         torch.save({
#             "encoder": encoder.state_dict(),
#             "decoder": decoder.state_dict(),
#         }, SAVE_PATH)

#     # ------------------ PLOT TRAINING LOSS ------------------
#     plt.figure()
#     plt.plot(all_epoch_losses)
#     plt.xlabel("Epoch")
#     plt.ylabel("Average REINFORCE Loss")
#     plt.title("Training Loss Curve")
#     plt.grid(True)
#     plt.savefig("training_loss.png")
#     plt.show()

#     print(f"\nTraining complete. Saved to {SAVE_PATH}")
#     print("Training loss plot saved as training_loss.png")

# # -----------------------------------------------------------
# # MAIN
# # -----------------------------------------------------------
# if __name__ == "__main__":
#     train_reinforce()
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# PATH FIX
# -----------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------
from env.topology import generate_topology
from env.sfc_request import generate_sfc
from env.simulator import compute_resource_variance
from models.transformer_encoder import TransformerEncoderModel
from models.transformer_decoder import TransformerDecoder

# -----------------------------------------------------------
# EXTRACT PATH NODE FEATURES
# -----------------------------------------------------------
def extract_path_node_features(G, path):
    feats = []
    for n in path:
        feats.append([
            float(G.nodes[n]['cpu']),
            float(G.nodes[n]['ram']),
            float(G.nodes[n]['energy'])
        ])
    return torch.tensor(feats, dtype=torch.float32)

# -----------------------------------------------------------
# REINFORCE LOSS
# -----------------------------------------------------------
def reinforce_loss(resource_variance, log_probs, gamma=1.0):
    total_log_prob = torch.stack(log_probs).sum()
    return gamma * resource_variance * total_log_prob

# -----------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------
def train_reinforce(
    num_epochs=100,
    sfc_per_epoch=10,
    samples_per_sfc=2,
    lr=0.0005,
    topology_nodes=20,
):

    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
    SAVE_PATH = os.path.join(PROJECT_ROOT, "experiments", "encoder_decoder.pth")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("Generating topology...")
    G = generate_topology(topology_nodes)

    encoder = TransformerEncoderModel(feature_dim=5, d_model=64)
    decoder = TransformerDecoder(d_model=64)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    print("Starting REINFORCE training...\n")
    all_epoch_losses = []

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

            optimizer.zero_grad()
            total_loss = 0.0

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
                total_loss += loss

            epoch_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / (sfc_per_epoch * samples_per_sfc)
        all_epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch} | Avg Loss = {avg_epoch_loss:.4f}")

        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        }, SAVE_PATH)

    print(f"\nTraining complete. Saved to {SAVE_PATH}")

    # ------------------------
    # PLOT TRAINING LOSS
    # ------------------------
    plt.figure()
    plt.plot(all_epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average REINFORCE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(PROJECT_ROOT, "plots/training_loss.png"))
    plt.show()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    train_reinforce()
