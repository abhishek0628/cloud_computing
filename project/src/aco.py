# src/aco.py
import numpy as np
import random
import torch
import torch.nn.functional as F

class AntColony:
    def __init__(self, env, node_logits=None, edge_logits=None, params=None, seed=None):
        self.env = env
        self.N = env.n
        self.rng = random.Random(seed)

        # pheromone matrix initialization
        self.tau = np.ones((self.N, self.N), dtype=np.float32) * 0.1

        # ACO hyperparameters
        params = params or {}
        self.alpha = params.get('alpha', 1.0)
        self.beta = params.get('beta', 2.0)
        self.gamma = params.get('gamma', 1.0)
        self.rho = params.get('rho', 0.1)
        self.Q = params.get('Q', 1.0)

        # Store Transformer priors if available
        if node_logits is not None:
            self.node_logits = node_logits.detach().clone()
            self.edge_logits = edge_logits.detach().clone()

            with torch.no_grad():
                self.node_prior_probs = F.softmax(self.node_logits, dim=0)
                self.edge_prior_probs = torch.sigmoid(self.edge_logits)
        else:
            self.node_logits = None
            self.edge_logits = None
            self.node_prior_probs = torch.ones(self.N) / self.N
            self.edge_prior_probs = torch.ones((self.N, self.N)) * 0.5

    # -------------------------------------------------------------------------
    # Heuristic function (classic ACO)
    # -------------------------------------------------------------------------
    def heuristic(self, u, v):
        G = self.env.graph
        if not G.has_edge(u, v):
            return 1e-6
        e = G.edges[u, v]
        return 1.0 / (e['delay'] + 1.0)

    # -------------------------------------------------------------------------
    # Transition probability for ACO (numpy version, rarely used now)
    # -------------------------------------------------------------------------
    def transition_probs_numpy(self, u, neighbors):
        probs = []
        for v in neighbors:
            tau = (self.tau[u, v] ** self.alpha)
            prior = (self.edge_prior_probs[u, v] ** self.beta)
            eta = (self.heuristic(u, v) ** self.gamma)
            probs.append(tau * prior * eta)
        probs = np.array(probs, dtype=np.float32)
        s = probs.sum()
        return (probs / (s + 1e-12)) if s > 0 else np.ones_like(probs) / len(probs)

    # -------------------------------------------------------------------------
    # Main colony solution construction using Torch sampling (for gradients)
    # -------------------------------------------------------------------------
    def construct_solution_for_sfc_torch(self, sfc, num_ants=10, topk_nodes=12, device='cpu'):
        """
        Construct candidate SFC solutions guided by Transformer priors.
        Returns a list of dicts with embedding, path, cost, and log_prob (torch tensor).
        """
        solutions = []
        N = self.N
        node_prior = self.node_prior_probs.to(device)
        edge_prior = self.edge_prior_probs.to(device)

        # precompute candidate nodes for each VNF
        node_scores = node_prior.cpu().numpy()
        node_idx_sorted = np.argsort(-node_scores)
        candidates_per_vnf = []

        for vnf in sfc.vnfs:
            cand = []
            for n in node_idx_sorted:
                node = self.env.graph.nodes[n]
                if node['cpu'] - node['load_cpu'] >= vnf.cpu and node['ram'] - node['load_ram'] >= vnf.ram:
                    cand.append(int(n))
                if len(cand) >= topk_nodes:
                    break
            if not cand:
                cand = [int(node_idx_sorted[0])]
            candidates_per_vnf.append(cand)

        # loop over ants
        for ant in range(num_ants):
            log_prob_acc = torch.tensor(0.0, device=device, requires_grad=True)
            embedding = []
            feasible = True

            # sample node for each VNF
            for vn_idx, cand in enumerate(candidates_per_vnf):
                cand = list(cand)
                probs = node_prior[cand]
                probs = probs / (probs.sum() + 1e-12)
                m = torch.distributions.Categorical(probs)
                idx = m.sample()
                chosen_node = cand[int(idx.item())]

                # accumulate log probability for gradient
                log_prob_acc = log_prob_acc + m.log_prob(idx)
                embedding.append(int(chosen_node))

            # build routing path
            path_nodes = []
            cur = sfc.src
            path_nodes.append(cur)

            for host in embedding:
                p = self.env.shortest_path(cur, host)
                if p is None:
                    feasible = False
                    break
                for node in p[1:]:
                    path_nodes.append(node)
                cur = host

            if not feasible:
                continue

            p = self.env.shortest_path(cur, sfc.dst)
            if p is None:
                continue
            for node in p[1:]:
                path_nodes.append(node)

            # validate resource constraints
            ok, total_delay, msg = self.env.check_embedding(path_nodes, embedding, sfc)
            if not ok:
                continue

            # compute total cost
            G = self.env.graph
            cost = 0.0
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                cost += G.edges[u, v]['cost'] * sfc.bandwidth
            for hn in set(embedding):
                cost += G.nodes[hn]['hosting_cost'] * 10.0

            # store solution
            solutions.append({
                'embedding': embedding,
                'path': path_nodes,
                'cost': float(cost),
                'log_prob': log_prob_acc  # ← FIXED: keep connected tensor
            })

        return solutions

    # -------------------------------------------------------------------------
    # Pheromone update rule (ACO classic)
    # -------------------------------------------------------------------------
    def update_pheromone(self, solutions):
        # Evaporate old pheromones
        self.tau *= (1.0 - self.rho)

        # Deposit new pheromone from each successful ant
        for sol in solutions:
            path = sol['path']
            cost = sol['cost']
            delta = self.Q / (cost + 1e-12)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.tau[u, v] += delta
                self.tau[v, u] += delta
