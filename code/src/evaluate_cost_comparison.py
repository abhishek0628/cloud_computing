import numpy as np
import matplotlib.pyplot as plt

# Example: assume you saved costs during evaluation
nn_costs = np.load("results/nn_costs.npy")
heuristic_costs = np.load("results/heuristic_costs.npy")

plt.plot(nn_costs, label="NN-based Deployment")
plt.plot(heuristic_costs, label="Heuristic Deployment")
plt.xlabel("SFC Index")
plt.ylabel("Deployment Cost")
plt.title("Cost Performance Comparison")
plt.legend()
plt.savefig("cost_comparison.png")
plt.show()
