# import torch

# # Path to your checkpoint
# pth_path = "encoder_decoder.pth"  # or include folder: "experiments/encoder_decoder.pth"

# # Load the checkpoint
# checkpoint = torch.load(pth_path, map_location='cpu')

# # Inspect what it contains
# print("Type of checkpoint:", type(checkpoint))

# if isinstance(checkpoint, dict):
#     print("Keys in the checkpoint:", checkpoint.keys())
# else:
#     print("Checkpoint might already be a full model object.")
import torch
import torch.nn as nn

# ------------------------------
# Replace these with your real model classes
# ------------------------------
class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ------------------------------
# Load the checkpoint
# ------------------------------
pth_path = "encoder_decoder.pth"
checkpoint = torch.load(pth_path, map_location='cpu')

print(f"Type of checkpoint: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
else:
    print("Checkpoint is not a dict. It might already be a full model object.")

# ------------------------------
# Initialize models
# ------------------------------
encoder = EncoderModel()
decoder = DecoderModel()

# ------------------------------
# Load weights
# ------------------------------
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Set models to evaluation mode
encoder.eval()
decoder.eval()

print("\nEncoder and decoder weights loaded successfully!")

# ------------------------------
# Dummy forward pass to verify
# ------------------------------
dummy_input = torch.randn(1, 10)  # Replace with correct input shape
encoded = encoder(dummy_input)
output = decoder(encoded)

print("\nDummy forward pass output:")
print(output)
