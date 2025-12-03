# import numpy as np
# import os

# # Parameters
# num_samples = 500       # Number of SFC requests
# seq_len = 5             # Number of VNFs per SFC
# num_features = 8        # Features per VNF (adjust according to your model)

# # Create folder if it doesn't exist
# os.makedirs("data", exist_ok=True)

# # Generate random SFC requests (values can be adjusted to your scenario)
# sfc_requests = np.random.rand(num_samples, seq_len, num_features).astype(np.float32)

# # Save to file
# np.save("data/sfc_requests.npy", sfc_requests)

# print("sfc_requests.npy generated in ./data folder")
# print("Shape:", sfc_requests.shape)
import numpy as np
import os

# Parameters (should match encoder)
num_samples = 500       # Number of SFC requests
seq_len = 5             # Number of VNFs per SFC
num_features = 8        # Features per VNF

# Create folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate random SFC requests (values can be adjusted)
sfc_requests = np.random.rand(num_samples, seq_len, num_features).astype(np.float32)

# Save to file
np.save("data/sfc_requests.npy", sfc_requests)

print("sfc_requests.npy generated in ./data folder")
print("Shape:", sfc_requests.shape)

# Example: preparing decoder input
# Typically decoder uses previous VNF embeddings + possibly SFC metadata
decoder_input = sfc_requests[:, :-1, :]  # all except last VNF
decoder_target = sfc_requests[:, 1:, :]  # next VNF as target

print("Decoder input shape:", decoder_input.shape)
print("Decoder target shape:", decoder_target.shape)

# Optional: save decoder inputs and targets separately
np.save("data/decoder_input.npy", decoder_input)
np.save("data/decoder_target.npy", decoder_target)
