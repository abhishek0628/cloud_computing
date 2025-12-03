import os
import random
import matplotlib.pyplot as plt

# -----------------------------
# 1. SIMULATED RL & RANDOM DATA
# (Replace this with your real values later)
# -----------------------------

num_episodes = 50

rl_rewards = []
random_rewards = []

for _ in range(num_episodes):
    rl_rewards.append(random.uniform(70, 100))      # RL performs better
    random_rewards.append(random.uniform(20, 60))   # Random performs worse

print("RL rewards:", rl_rewards[:5], "...")
print("Random rewards:", random_rewards[:5], "...")

# -----------------------------
# 2. CREATE PLOTS DIRECTORY
# -----------------------------

BASE_DIR = os.getcwd()
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# 3. BOX PLOT: RL VS RANDOM
# -----------------------------

save_path = os.path.join(PLOT_DIR, "rl_vs_random.png")

plt.figure()
plt.boxplot([rl_rewards, random_rewards], labels=["RL", "Random"])
plt.title("RL vs Random Performance")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print("✅ Plot saved at:", save_path)
