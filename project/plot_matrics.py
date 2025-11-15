import pickle
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Path to your .pkl file (adjust if you used a different experiment name)
# ---------------------------------------------------------------------
pkl_path = "experiments/66/train_metrics.pkl"

# ---------------------------------------------------------------------
# Load the pickle file
# ---------------------------------------------------------------------
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"File not found: {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# ---------------------------------------------------------------------
# Inspect available keys
# ---------------------------------------------------------------------
print("Available keys in file:", data.keys())
print("First few epochs:", data["epoch"][:10])
print("First few costs:", data["cost"][:10])
print("First few baselines:", data["baseline"][:10])
print("First few success rates:", data["success_rate"][:10])

# ---------------------------------------------------------------------
# Plot 1: Cost vs Epoch
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(data["epoch"], data["cost"], label="SFC Deployment Cost", color="blue")
plt.plot(data["epoch"], data["baseline"], label="Baseline Cost", color="orange", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training Convergence Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Plot 2: Success Rate vs Epoch
# ---------------------------------------------------------------------
if "success_rate" in data:
    plt.figure(figsize=(8, 5))
    plt.plot(data["epoch"], data["success_rate"], label="Success Rate", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.title("Service Function Chain (SFC) Success Rate Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nPlots generated successfully!")
