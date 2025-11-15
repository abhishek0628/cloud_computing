import pickle

with open("experiments/66/train_metrics.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["epoch"][:10])  # print first 10 epochs
