# minimal plotting code snippet to add at bottom of your eval.py
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import torch
from src.models.transformer_encoder import SmallGraphTransformer
from src.env import NTNEnv
import numpy as np
import os

def plot_convergence(metrics_path, out_dir='out'):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    epochs = metrics['epoch']
    costs = metrics['cost']
    baseline = metrics['baseline']
    plt.figure(figsize=(8,4))
    plt.plot(epochs, costs, label='cost')
    plt.plot(epochs, baseline, label='baseline', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'convergence.png'))
    print("Saved", os.path.join(out_dir, 'convergence.png'))

def plot_heatmap(env, model, out_path):
    node_feats, _ = env.get_features()
    device = next(model.parameters()).device
    node_feats_t = torch.from_numpy(node_feats).to(device)
    with torch.no_grad():
        node_logits, edge_logits = model(node_feats_t)
        edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
    G = env.graph
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8,8))
    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=40)
    # draw edges with width proportional to edge_probs
    for u,v in G.edges():
        w = edge_probs[u,v]
        if w < 0.01: continue
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], width=1 + 4*w, alpha=0.7)
    plt.axis('off')
    plt.savefig(out_path)
    print("Saved", out_path)
