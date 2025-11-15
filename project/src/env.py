import random
import networkx as nx
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class VNF:
    cpu: int
    ram: int
    bw: int
    proc_delay: float
    dur: float

@dataclass
class SFCRequest:
    src: int
    dst: int
    vnfs: List[VNF]
    bandwidth: int
    max_delay: float
    id: int = 0

class NTNEnv:
    def __init__(self, n_nodes=66, k=4, seed=None):
        self.seed = seed
        self.n = n_nodes
        self.graph = None
        self.rng = random.Random(seed)
        self._build_graph(k)

    def _build_graph(self, k):
        G = nx.random_geometric_graph(self.n, radius=0.3, seed=self.seed)
        # ensure connectivity: add k-NN edges if needed
        pos = nx.get_node_attributes(G, 'pos')
        # compute k nearest neighbors by Euclidean distance
        nodes = list(G.nodes())
        for i in nodes:
            dists = []
            xi, yi = pos[i]
            for j in nodes:
                if i == j: continue
                xj, yj = pos[j]
                dists.append((np.hypot(xi-xj, yi-yj), j))
            dists.sort()
            for _, j in dists[:k]:
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
        # assign attributes
        for u in G.nodes():
            G.nodes[u]['cpu'] = int(self.rng.randint(100, 300))
            G.nodes[u]['ram'] = int(self.rng.randint(100, 300))
            G.nodes[u]['energy'] = int(self.rng.randint(1000, 5000))
            G.nodes[u]['hosting_cost'] = float(self.rng.uniform(0.5, 2.0))
            G.nodes[u]['load_cpu'] = 0
            G.nodes[u]['load_ram'] = 0
        for u,v in G.edges():
            G.edges[u,v]['bandwidth'] = int(self.rng.randint(100, 1000))
            G.edges[u,v]['delay'] = float(self.rng.uniform(5.0, 40.0))  # ms
            G.edges[u,v]['cost'] = float(self.rng.uniform(0.1, 1.0))
            G.edges[u,v]['active'] = True
        self.graph = G

    def sample_sfc(self, max_chain=4, max_cpu=80, max_bw=50):
        src = self.rng.randrange(self.n)
        dst = self.rng.randrange(self.n)
        while dst == src:
            dst = self.rng.randrange(self.n)
        chain_len = self.rng.randint(1, max_chain)
        vnfs = []
        for _ in range(chain_len):
            vnfs.append(VNF(cpu=self.rng.randint(10, max_cpu),
                            ram=self.rng.randint(8, 64),
                            bw=self.rng.randint(10, max_bw),
                            proc_delay=self.rng.uniform(1.0, 5.0),
                            dur=self.rng.uniform(5.0, 50.0)))
        max_delay = chain_len * 200.0  # generous
        req = SFCRequest(src=src, dst=dst, vnfs=vnfs, bandwidth=self.rng.randint(10, max_bw), max_delay=max_delay)
        return req

    def get_features(self):
        # returns node feature matrix (N x d) and adjacency edge features (N x N with masked non-edges)
        N = self.n
        node_feats = []
        for i in range(N):
            n = self.graph.nodes[i]
            node_feats.append([
                n['cpu'], n['ram'], n['energy'],
                n['hosting_cost'], n['load_cpu'], n['load_ram']
            ])
        node_feats = np.array(node_feats, dtype=np.float32)
        # normalize roughly
        node_feats[:,0] /= 300.0
        node_feats[:,1] /= 300.0
        node_feats[:,2] /= 5000.0
        node_feats[:,4] /= 300.0
        node_feats[:,5] /= 300.0
        # adjacency edge features matrix
        adj_delay = np.zeros((N,N), dtype=np.float32)
        adj_bw = np.zeros((N,N), dtype=np.float32)
        adj_cost = np.zeros((N,N), dtype=np.float32)
        for u,v,edata in self.graph.edges(data=True):
            adj_delay[u,v] = edata['delay']
            adj_delay[v,u] = edata['delay']
            adj_bw[u,v] = edata['bandwidth']
            adj_bw[v,u] = edata['bandwidth']
            adj_cost[u,v] = edata['cost']
            adj_cost[v,u] = edata['cost']
        # normalize edge features
        adj_delay = adj_delay / 100.0
        adj_bw = adj_bw / 1000.0
        adj_cost = adj_cost / 2.0
        edge_feats = np.stack([adj_delay, adj_bw, adj_cost], axis=-1)  # N x N x 3
        return node_feats, edge_feats

    def reset_loads(self):
        for i in self.graph.nodes():
            self.graph.nodes[i]['load_cpu'] = 0
            self.graph.nodes[i]['load_ram'] = 0

    def check_embedding(self, path_nodes, embedding_assignments, sfc:SFCRequest):
        # embedding_assignments: list of node ids where each VNF is placed, len = m
        # path_nodes: list of nodes forming path covering those nodes in order (simple validation)
        # returns (success:bool, total_delay, violation_str)
        G = self.graph
        # check capacities
        for vn_idx, node in enumerate(embedding_assignments):
            vnf = sfc.vnfs[vn_idx]
            node_cpu = G.nodes[node]['cpu'] - G.nodes[node]['load_cpu']
            node_ram = G.nodes[node]['ram'] - G.nodes[node]['load_ram']
            if vnf.cpu > node_cpu or vnf.ram > node_ram:
                return False, None, f"node {node} resource insufficient for vnf {vn_idx}"
        # compute delay across path (simple sum of link delays + proc)
        total = 0.0
        # path must include the embed nodes in order - we assume path_nodes is path between src->dst covering embeddings
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if not G.has_edge(u,v):
                return False, None, f"missing edge {u}-{v}"
            total += G.edges[u,v]['delay']
        for vnf in sfc.vnfs:
            total += vnf.proc_delay
        if total > sfc.max_delay:
            return False, total, "delay exceed"
        return True, total, "ok"

    def apply_embedding(self, embedding_assignments, sfc:SFCRequest):
        # consume resource on assigned nodes and links by bandwidth
        G = self.graph
        for vn_idx, node in enumerate(embedding_assignments):
            vnf = sfc.vnfs[vn_idx]
            G.nodes[node]['load_cpu'] += vnf.cpu
            G.nodes[node]['load_ram'] += vnf.ram

    def shortest_path(self, src, dst):
        try:
            p = nx.shortest_path(self.graph, src, dst, weight='delay')
            return p
        except nx.NetworkXNoPath:
            return None

    def k_shortest_paths(self, src, dst, k=3):
        # simple wrapper: use networkx all_shortest_paths (not k-shortest robust, but fine)
        paths = []
        try:
            for p in nx.all_shortest_paths(self.graph, source=src, target=dst, weight='delay'):
                paths.append(p)
                if len(paths) >= k:
                    break
        except nx.exception.NetworkXNoPath:
            pass
        if not paths:
            p = self.shortest_path(src, dst)
            if p: paths.append(p)
        return paths
