import networkx as nx
import random

def generate_topology(n_nodes=40):
    G = nx.DiGraph()

    # Add nodes with resources
    for i in range(n_nodes):
        G.add_node(i, 
                   cpu=random.randint(5,15),
                   ram=random.randint(5,15),
                   energy=random.randint(5,15))
    
    # Add edges with features
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and random.random() < 0.15:
                G.add_edge(i, j,
                           delay=random.uniform(1,10),
                           bandwidth=random.uniform(5,20),
                           duration=random.uniform(10,30))
    return G
