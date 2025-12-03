# src/eval/evaluate.py
import numpy as np
import torch
from src.utils.data_loader import load_topology, load_sfc_requests
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from code.src.models.gnn_heuristicc import GNNHeuristic
from src.aco.aco import ACO

def evaluate(transformer_model_path, gnn_model_path, topology_file, sfc_file):
    # ---------- Load topology ----------
    G = load_topology(topology_file)
    sfc_requests = load_sfc_requests(sfc_file)

    # ---------- Load trained models ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    gnn = GNNHeuristic().to(device)

    # Load checkpoints
    encoder.load_state_dict(torch.load(transformer_model_path)["encoder"])
    decoder.load_state_dict(torch.load(transformer_model_path)["decoder"])
    gnn.load_state_dict(torch.load(gnn_model_path))
    
    encoder.eval()
    decoder.eval()
    gnn.eval()

    # ---------- Initialize ACO ----------
    aco = ACO(graph=G)

    # ---------- Matrices to collect ----------
    vnf_cost_list = []
    sfc_cost_list = []
    heuristic_trained = np.zeros((len(G.nodes), len(G.nodes)))
    cost_comparison = []
    success_rate_list = []
    delay_list = []

    # ---------- Evaluate each SFC request ----------
    for sfc in sfc_requests:
        # 1. Run encoder-decoder for VNF embedding
        with torch.no_grad():
            # Example: get VNF embedding cost (variance)
            vnf_cost = np.random.randint(0, 35000)  # placeholder, replace with actual output
            vnf_cost_list.append(vnf_cost)

        # 2. Run GNN heuristic module to get SFC deployment cost
        with torch.no_grad():
            sfc_cost = np.random.uniform(5, 80)  # placeholder, replace with actual output
            sfc_cost_list.append(sfc_cost)

        # 3. Generate heuristic matrix
        # Placeholder: random paths
        heuristic_trained[np.random.randint(0,len(G.nodes)), np.random.randint(0,len(G.nodes))] = 1.0

        # 4. Run Transformer-ACO for routing and embedding
        solution = aco.run(sfc_request=sfc, encoder=encoder, decoder=decoder, heuristic=gnn)
        cost_comparison.append(solution["cost"])
        success_rate_list.append(solution["success"])
        delay_list.append(solution["delay"])

    # ---------- Save matrices ----------
    np.save("experiments/plots/vnf_cost.npy", np.array(vnf_cost_list))
    np.save("experiments/plots/sfc_cost.npy", np.array(sfc_cost_list))
    np.save("experiments/plots/heuristic_trained.npy", heuristic_trained)
    np.save("experiments/plots/cost_comparison.npy", np.array(cost_comparison))
    np.save("experiments/plots/success_delay.npy", np.vstack([success_rate_list, delay_list]))

    print("Evaluation complete! Matrices saved in experiments/plots/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_model", type=str, default="checkpoints/encoder_decoder.pth")
    parser.add_argument("--gnn_model", type=str, default="checkpoints/gnn_heuristic.pth")
    parser.add_argument("--topology", type=str, default="data/topology/walker66.gml")
    parser.add_argument("--sfc_requests", type=str, default="data/sfc_requests.npy")
    args = parser.parse_args()

    evaluate(args.transformer_model, args.gnn_model, args.topology, args.sfc_requests)
