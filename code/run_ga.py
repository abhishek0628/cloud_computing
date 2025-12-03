# run_ga.py
from models.ga_sfc import GASFC
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_vnfs = 5
num_nodes = 10
pop_size = 50
generations = 50
mutation_rate = 0.1

# Initialize GA model
ga_model = GASFC(num_vnfs, num_nodes, pop_size, generations, mutation_rate)

# Run GA
best_deployment, best_cost, cost_history = ga_model.run()

# Print results
print("Best deployment:", best_deployment)
print("Best cost:", best_cost)

# Save results
np.save("results/ga_deployment.npy", best_deployment)
np.save("results/ga_cost.npy", best_cost)

# Plot convergence
plt.plot(cost_history)
plt.xlabel("Generation")
plt.ylabel("Deployment Cost")
plt.title("GA Convergence")
plt.savefig("plots/ga_convergence.png")
plt.show()
