import random
import numpy as np

class ACO:
    def __init__(self, graph, alpha=1, beta=2, rho=0.1, n_ants=20):
        self.G = graph
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.tau = { (i,j):1 for i,j in graph.edges }
    
    def heuristic(self, i, j):
        return 1 / (self.G[i][j]['delay'] + 1e-6)
    
    def choose(self, curr, allowed):
        probs = []
        edges = []

        for j in allowed:
            tau = self.tau.get((curr,j), 1)
            eta = self.heuristic(curr, j)
            probs.append((tau**self.alpha)*(eta**self.beta))
            edges.append(j)

        probs = np.array(probs)
        probs /= probs.sum()

        return np.random.choice(edges, p=probs)
    
    def find_path(self, src, dst):
        for _ in range(self.n_ants):
            curr = src
            visited = [curr]
            while curr != dst:
                neighbors = list(self.G[curr])
                neighbors = [n for n in neighbors if n not in visited]
                if not neighbors:
                    break
                curr = self.choose(curr, neighbors)
                visited.append(curr)
            if visited[-1] == dst:
                return visited
        return None
