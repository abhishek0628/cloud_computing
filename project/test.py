import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
isa = ["PISA", "ALPHA"]

# LLC Miss Rate (example values)
lru_miss = [0.9864, 0.7014]
lip_miss = [0.9864, 0.7014]  # PISA uses LIP_9, Alpha uses LIP_5

# MPKI (example values)
lru_mpki = [14.163, 13.457]
lip_mpki = [14.163, 13.457]

# IPC (example values)
lru_ipc = [0.2506, 0.2610]
lip_ipc = [0.2506, 0.2610]

# --- Create grouped bar positions ---
x = np.arange(len(isa))
width = 0.35

# --- 1. LLC Miss Rate ---
# plt.figure(figsize=(6,4))
# plt.bar(x - width/2, lru_miss, width, label='LRU', color='skyblue')
# plt.bar(x + width/2, lip_miss, width, label='LIP_9 (PISA) / LIP_5 (ALPHA)', color='salmon')
# plt.xticks(x, isa)
# plt.ylabel('Miss Rate')
# plt.title('LLC Miss Rate Comparison')
# plt.legend()
# plt.tight_layout()
# plt.savefig("llc_miss_rate_comparison.png")
# plt.show()

# --- 2. MPKI ---
# plt.figure(figsize=(6,4))
# plt.bar(x - width/2, lru_mpki, width, label='LRU', color='skyblue')
# plt.bar(x + width/2, lip_mpki, width, label='LIP_9 (PISA) / LIP_5 (ALPHA)', color='salmon')
# plt.xticks(x, isa)
# plt.ylabel('MPKI')
# plt.title('LLC MPKI Comparison')
# plt.legend()
# plt.tight_layout()
# plt.savefig("mpki_comparison.png")
# plt.show()

# --- 3. IPC ---
plt.figure(figsize=(6,4))
plt.bar(x - width/2, lru_ipc, width, label='LRU', color='skyblue')
plt.bar(x + width/2, lip_ipc, width, label='LIP_9 (PISA) / LIP_5 (ALPHA)', color='salmon')
plt.xticks(x, isa)
plt.ylabel('IPC')
plt.title('IPC Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("ipc_comparison.png")
plt.show()
