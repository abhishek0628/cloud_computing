# src/trainer.py
import argparse
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from src.env import NTNEnv
from src.models.transformer_encoder import SmallGraphTransformer
from src.aco import AntColony
from src.utils import set_seed, make_dirs, save_model
import os
import math

def run_training(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create three topology sizes to reproduce paper (66, 144, 192)
    topo_sizes = [66]  # you can iterate and run separate experiments for 144 and 192; do one at a time to save RAM
    env = NTNEnv(n_nodes=topo_sizes[0], k=4, seed=args.seed)

    model = SmallGraphTransformer(node_feat_dim=6, d_model=64, nhead=4, nlayers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    baseline = None  # running baseline per paper b(r)
    baseline_momentum = 0.99

    best_val = 1e18
    make_dirs('experiments')

    metrics = {'epoch': [], 'cost': [], 'baseline': [], 'success_rate': []}

    for epoch in trange(args.epochs, desc="Epochs"):
        # sample one SFC request per epoch (paper seems per-request training)
        sfc = env.sample_sfc(max_chain=args.max_chain)
        # prepare features
        node_feats, edge_feats = env.get_features()
        node_feats_t = torch.from_numpy(node_feats).to(device)
        # forward pass to get priors
        node_logits, edge_logits = model(node_feats_t)  # tensors on device
        # run ACO colony guided by priors
        aco = AntColony(env, node_logits=node_logits, edge_logits=edge_logits, params={'alpha':args.alpha,'beta':args.beta,'gamma':args.gamma,'rho':args.rho,'Q':args.Q}, seed=args.seed)

        # run ants (torch-sampling) and collect solutions with log_prob
        solutions = aco.construct_solution_for_sfc_torch(sfc, num_ants=args.num_ants, topk_nodes=args.topk, device=device)

        if not solutions:
            # no feasible solution found
            epoch_cost = 1e5
            success = 0
            # small negative learning signal: encourage exploration by applying small penalty
            # we set loss = (cost - baseline) * (-small_value) to nudge model
            # but better: skip update (or apply gradient to encourage exploring different priors)
            # we'll still update baseline
            log_prob_tensor = torch.tensor(0.0, device=device)
        else:
            # pick the best solution among ants (min cost)
            costs = np.array([s['cost'] for s in solutions], dtype=np.float32)
            best_idx = int(np.argmin(costs))
            best_sol = solutions[best_idx]
            epoch_cost = float(best_sol['cost'])
            log_prob_tensor = best_sol['log_prob']  # torch scalar on device
            success = 1

        # update baseline (exponential moving average)
        if baseline is None:
            baseline = epoch_cost
        else:
            baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * epoch_cost

        advantage = epoch_cost - baseline  # positive if worse than baseline
        # REINFORCE loss (minimize cost): we want to reduce probability of choices that lead to higher cost
        # Loss = advantage * log_prob  (since log_prob is negative for low-prob actions)
        # However to push probabilities to produce lower cost we use loss = advantage * log_prob_tensor
        # We detach advantage (float) before making tensor
        adv_t = torch.tensor(advantage, dtype=torch.float32, device=device)
        loss = adv_t * log_prob_tensor  # shape scalar

        # optimization step:
        optimizer.zero_grad()
        # in case log_prob_tensor == 0 (no solution), loss will be zero -> no update
        loss.backward(retain_graph=False)
        optimizer.step()

        # update pheromones with all solutions in colony (paper updates from ants)
        if solutions:
            aco.update_pheromone(solutions)

        # logging & save
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch} cost={epoch_cost:.3f} baseline={baseline:.3f} adv={advantage:.3f} success={success}")
        metrics['epoch'].append(epoch)
        metrics['cost'].append(epoch_cost)
        metrics['baseline'].append(baseline)
        metrics['success_rate'].append(success)

        if epoch_cost < best_val:
            best_val = epoch_cost
            save_model(model, os.path.join('experiments', 'best_model.pth'))

    # save final metrics to disk for plotting
    import pickle
    with open(os.path.join(args.out, 'train_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    print("Training done. Best cost:", best_val)
    print("Metrics saved to", os.path.join(args.out, 'train_metrics.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_ants', type=int, default=32)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='experiments')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--Q', type=float, default=1.0)
    parser.add_argument('--max_chain', type=int, default=3)
    parser.add_argument('--log_every', type=int, default=10)
    args = parser.parse_args()
    run_training(args)
