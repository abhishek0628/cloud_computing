import pickle

with open("experiments/66/train_metrics.pkl", "rb") as f:
    data = pickle.load(f)

print("Available keys:", data.keys())
print("First 10 epochs:", data["epoch"][:10])
print("First 10 costs:", data["cost"][:10])
