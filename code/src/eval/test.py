import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Create directory to save plots
# ----------------------------
os.makedirs("plots", exist_ok=True)

# ----------------------------
# Metric logs
# ----------------------------
vnf_variance_log = []
sfc_cost_log = []
heuristic_untrained_log = []
heuristic_trained_log = []
cost_comparison_log = {}   # algorithm -> list of costs
success_rate_log = {}      # algorithm -> success rate
avg_delay_log = {}         # algorithm -> average delay

# ----------------------------
# Example network and algorithms
# Replace with your actual network and SFC environment
# ----------------------------
network = None   # your network object
algorithms = ["Greedy", "DeepLearning", "Transformer-ACO", "Proposed"]
sfc_requests = None  # your SFC request list

# ----------------------------
# TRAINING LOOP (Transformer + GNN)
# ----------------------------
num_epochs = 20
for epoch in range(num_epochs):
    # --- Transformer embedding ---
    # Replace with your actual prediction
    predicted_paths = []  # compute predicted VNF paths
    actual_paths = []     # ground truth paths

    # Compute embedding cost / variance
    vnf_cost = np.random.rand()  # <-- replace with real cost calculation
    vnf_variance_log.append(vnf_cost)

    # --- GNN SFC deployment ---
    # Deploy SFCs using GNN/heuristic
    sfc_cost = np.random.rand()  # <-- replace with actual deployment cost
    sfc_cost_log.append(sfc_cost)

# ----------------------------
# HEURISTIC FUNCTION VISUALIZATION
# ----------------------------
edges = range(1, 51)  # your network edges
# Before training
heuristic_untrained_log = np.random.rand(len(edges))  # replace with actual heuristic scores
# After training
heuristic_trained_log = np.random.rand(len(edges)) + 0.5  # replace with actual trained scores

# ----------------------------
# COST PERFORMANCE COMPARISON
# ----------------------------
num_trials = 10
for algo in algorithms:
    costs = []
    for _ in range(num_trials):
        cost = np.random.rand() * 5  # replace with real algorithm evaluation
        costs.append(cost)
    cost_comparison_log[algo] = costs

# ----------------------------
# SUCCESS RATE & DELAY PERFORMANCE
# ----------------------------
for algo in algorithms:
    successful_requests = np.random.randint(7, 10)  # replace with real count
    total_requests = 10
    delays = np.random.rand(total_requests) * 10     # replace with real delays
    success_rate_log[algo] = successful_requests / total_requests
    avg_delay_log[algo] = np.mean(delays)

# ----------------------------
# PLOTTING
# ----------------------------

# 1. VNF Embedding Cost
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), vnf_variance_log, marker='o', color='blue')
plt.title("Convergence of VNF Embedding Cost")
plt.xlabel("Epoch")
plt.ylabel("Resource Variance")
plt.savefig("plots/vnf_embedding_convergence.png")
plt.close()

# 2. SFC Deployment Cost
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), sfc_cost_log, marker='s', color='green')
plt.title("Convergence of SFC Deployment Cost")
plt.xlabel("Epoch")
plt.ylabel("Deployment Cost")
plt.savefig("plots/sfc_deployment_convergence.png")
plt.close()

# 3. Heuristic Function Visualization
plt.figure(figsize=(8,5))
plt.plot(edges, heuristic_untrained_log, label="Untrained", marker='o')
plt.plot(edges, heuristic_trained_log, label="Trained", marker='s')
plt.title("Heuristic Function Visualization")
plt.xlabel("Edge Index")
plt.ylabel("Heuristic Score η(i,j)")
plt.legend()
plt.savefig("plots/heuristic_visualization.png")
plt.close()

# 4. Cost Performance Comparison
cost_means = [np.mean(cost_comparison_log[a]) for a in algorithms]
cost_std = [np.std(cost_comparison_log[a]) for a in algorithms]

plt.figure(figsize=(8,5))
plt.bar(algorithms, cost_means, yerr=cost_std, capsize=5, color='orange')
plt.title("Cost Performance Comparison")
plt.ylabel("Average Cost")
plt.savefig("plots/cost_comparison.png")
plt.close()

# 5. Success Rate & Delay Performance
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel('Algorithm')
ax1.set_ylabel('Success Rate', color=color)
ax1.bar(algorithms, [success_rate_log[a] for a in algorithms], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Delay', color=color)
ax2.plot(algorithms, [avg_delay_log[a] for a in algorithms], color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Success Rate and Delay Performance")
fig.tight_layout()
plt.savefig("plots/success_delay_performance.png")
plt.close()

print("All plots generated and saved in ./plots/")
