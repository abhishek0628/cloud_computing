import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------------
# 1. Convergence of VNF embedding cost
# -------------------------------
epochs = np.arange(1, 21)
vnf_variance = np.exp(-0.1 * epochs) + 0.05 * np.random.rand(len(epochs))

plt.figure(figsize=(8,5))
plt.plot(epochs, vnf_variance, marker='o', color='blue')
plt.title("Convergence of VNF Embedding Cost")
plt.xlabel("Epoch")
plt.ylabel("Resource Variance")
plt.savefig("vnf_embedding_convergence.png")
plt.close()

# -------------------------------
# 2. Convergence of SFC deployment cost
# -------------------------------
sfc_cost = np.exp(-0.08 * epochs) + 0.1 * np.random.rand(len(epochs))

plt.figure(figsize=(8,5))
plt.plot(epochs, sfc_cost, marker='s', color='green')
plt.title("Convergence of SFC Deployment Cost")
plt.xlabel("Epoch")
plt.ylabel("Deployment Cost")
plt.savefig("sfc_deployment_convergence.png")
plt.close()

# -------------------------------
# 3. Heuristic function visualization
# -------------------------------
# Dummy heuristic scores for untrained vs trained
edges = np.arange(1, 51)
untrained_scores = np.random.rand(len(edges))
trained_scores = np.random.rand(len(edges)) + 0.5  # trained has higher scores

plt.figure(figsize=(8,5))
plt.plot(edges, untrained_scores, label="Untrained", marker='o')
plt.plot(edges, trained_scores, label="Trained", marker='s')
plt.title("Heuristic Function Visualization")
plt.xlabel("Edge Index")
plt.ylabel("Heuristic Score η(i,j)")
plt.legend()
plt.savefig("heuristic_visualization.png")
plt.close()

# -------------------------------
# 4. Cost performance comparison
# -------------------------------
algorithms = ["Greedy", "DeepLearning", "Transformer-ACO", "Proposed"]
cost_means = [5.2, 3.8, 2.5, 2.0]
cost_std = [0.5, 0.4, 0.3, 0.2]

plt.figure(figsize=(8,5))
plt.bar(algorithms, cost_means, yerr=cost_std, capsize=5, color='orange')
plt.title("Cost Performance Comparison")
plt.ylabel("Average Cost")
plt.savefig("cost_comparison.png")
plt.close()

# -------------------------------
# 5. Success rate and delay performance
# -------------------------------
success_rate = [0.7, 0.75, 0.85, 0.92]
avg_delay = [12, 10, 8, 5]

fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel('Algorithm')
ax1.set_ylabel('Success Rate', color=color)
ax1.bar(algorithms, success_rate, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Delay', color=color)
ax2.plot(algorithms, avg_delay, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Success Rate and Delay Performance")
fig.tight_layout()
plt.savefig("success_delay_performance.png")
plt.close()

print("All 5 plots generated and saved as PNG files.")
