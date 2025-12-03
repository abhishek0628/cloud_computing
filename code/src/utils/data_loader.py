# src/utils/data_loader.py
import networkx as nx
import numpy as np

def load_topology(file_path):
    """
    Load the network topology from a file.
    Supports GML, GraphML, or edge list formats.
    Returns a NetworkX graph.
    """
    if file_path.endswith(".gml"):
        G = nx.read_gml(file_path)
    elif file_path.endswith(".graphml"):
        G = nx.read_graphml(file_path)
    elif file_path.endswith(".edgelist"):
        G = nx.read_edgelist(file_path)
    else:
        raise ValueError("Unsupported topology file format.")
    return G

def load_sfc_requests(file_path):
    """
    Load SFC requests.
    Each request could include VNFs, required resources, source, and destination nodes.
    Returns a list of dictionaries.
    """
    # Example: load from numpy or CSV
    try:
        sfc_data = np.load(file_path, allow_pickle=True)
        sfc_requests = sfc_data.tolist()
    except:
        raise FileNotFoundError(f"Cannot load SFC requests from {file_path}")
    return sfc_requests
