import networkx as nx
import numpy as np

class SFCEnv:
    """
    Environment for Service Function Chain (SFC) request routing and VNF embedding.
    """
    def __init__(self, topology_file):
        """
        Initialize the environment.
        :param topology_file: Path to a network topology file (GML, GraphML, etc.)
        """
        self.graph = nx.read_gml(topology_file)
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        self.reset()

    def reset(self):
        """
        Reset environment state.
        """
        # Keep track of resource distribution for each node
        self.node_resources = {node: self.graph.nodes[node].get('capacity', 100) for node in self.nodes}
        # Keep track of used VNFs
        self.embedded_vnfs = {}
        # Reset current path
        self.current_path = []
        return self._get_state()

    def _get_state(self):
        """
        Return the current state representation.
        For simplicity, returns node resource distribution.
        """
        return np.array([self.node_resources[node] for node in self.nodes])

    def step(self, node, vnf_demand=1):
        """
        Take an action: embed a VNF at a node.
        :param node: Node ID where VNF is to be embedded.
        :param vnf_demand: Resource demand of the VNF.
        :return: state, cost, done
        """
        if self.node_resources[node] >= vnf_demand:
            self.node_resources[node] -= vnf_demand
            self.embedded_vnfs[node] = self.embedded_vnfs.get(node, 0) + 1
            cost = vnf_demand  # Simple cost: resource used
        else:
            cost = float('inf')  # Cannot embed VNF here

        done = False
        if sum(self.node_resources.values()) == 0 or len(self.embedded_vnfs) >= self.num_nodes:
            done = True

        state = self._get_state()
        return state, cost, done

    def get_possible_actions(self):
        """
        Return list of nodes where VNF can be embedded (resource available).
        """
        return [node for node in self.nodes if self.node_resources[node] > 0]

    def compute_path_cost(self, path):
        """
        Compute cost of a routing path.
        For example, sum of inverse remaining resources.
        """
        cost = 0
        for node in path:
            res = self.node_resources.get(node, 0)
            if res > 0:
                cost += 1 / res
            else:
                cost += 1e6  # penalize overloaded node
        return cost
