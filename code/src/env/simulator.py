import numpy as np

def compute_path_delay(G, path):
    """Return total delay along the path."""
    total_delay = 0.0
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        if G.has_edge(u, v):
            total_delay += G[u][v]['delay']
        else:
            return float('inf')   # invalid path
    
    return total_delay


def compute_bandwidth_cost(G, path, sfc):
    """Compute bandwidth cost based on edge capacities."""
    total_bw_cost = 0.0
    
    # max bw requirement among all VNFs
    bw_req = max(vnf['bw'] for vnf in sfc)
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        if not G.has_edge(u, v):
            return float('inf')
        
        edge_bw = G[u][v]['bandwidth']
        
        # If VNF BW > link BW, not feasible
        if bw_req > edge_bw:
            return float('inf')
        
        # Penalize low-bandwidth edges
        total_bw_cost += bw_req / (edge_bw + 1e-6)
    
    return total_bw_cost


def check_embedding_feasibility(G, path, sfc):
    """
    Check if VNFs can be placed on nodes along the path.
    For simplicity: place VNF[k] on node path[k % len(path)].
    You will replace this with Transformer decoder later.
    """
    node_resources = {n: {
        'cpu': G.nodes[n]['cpu'],
        'ram': G.nodes[n]['ram'],
        'energy': G.nodes[n]['energy']
    } for n in path}
    
    for k, vnf in enumerate(sfc):
        node = path[k % len(path)]  # simple round-robin placement
        
        if (node_resources[node]['cpu'] < vnf['cpu'] or
            node_resources[node]['ram'] < vnf['ram'] or
            node_resources[node]['energy'] < vnf['energy']):
            return False  # Not enough resources
        
        # Deduct resources
        node_resources[node]['cpu'] -= vnf['cpu']
        node_resources[node]['ram'] -= vnf['ram']
        node_resources[node]['energy'] -= vnf['energy']
    
    return True


def compute_resource_variance(G, path):
    """Variance of node CPU along path → used in loss function."""
    cpus = [G.nodes[n]['cpu'] for n in path]
    if len(cpus) == 0:
        return float('inf')
    return np.var(cpus)


def evaluate_solution(G, path, sfc, alpha=1.0, beta=0.3, gamma=0.3):
    """
    Return:
        cost, feasible_flag, total_delay, bandwidth_cost, resource_variance
    """
    if path is None or len(path) < 2:
        return float('inf'), False, None, None, None

    delay = compute_path_delay(G, path)
    bw_cost = compute_bandwidth_cost(G, path, sfc)
    var_res = compute_resource_variance(G, path)
    feasible_embed = check_embedding_feasibility(G, path, sfc)

    # If embedding is not possible, mark infeasible
    if not feasible_embed:
        return float('inf'), False, delay, bw_cost, var_res

    # Total cost
    cost = alpha*delay + beta*bw_cost + gamma*var_res

    return cost, True, delay, bw_cost, var_res
