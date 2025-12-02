# import torch
# import torch.nn as nn

# # Define your model architecture
# checkpoint = torch.load("experiments/encoder_decoder.pth", map_location='cpu')

# # If it's a state_dict:
# print(checkpoint.keys())  # see what keys are in the checkpoint
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc1 = nn.Linear(10, 50)
#         self.fc2 = nn.Linear(50, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# model = MyModel()

# # Load the saved weights
# model.load_state_dict(torch.load("encoder_decoder.pth"))

# # Set the model to evaluation mode (important for inference)
# model.eval()

import torch

# Path to your .pth file
pth_path = "experiments/encoder_decoder.pth"

# Load the checkpoint
checkpoint = torch.load(pth_path, map_location='cpu')

# Inspect what type of object it is
print(f"Type of checkpoint: {type(checkpoint)}\n")

# If it's a dictionary, list its keys
if isinstance(checkpoint, dict):
    print("Keys in the checkpoint:")
    for key in checkpoint.keys():
        print(" -", key)
else:
    print("Checkpoint is not a dict. It might be a full model object or a state_dict.")

# Example: if it contains 'model_state_dict', you can load it into your model
# Replace 'YourEncoderDecoderModel' with the class used during training
try:
    if 'model_state_dict' in checkpoint:
        from your_model_file import YourEncoderDecoderModel  # replace this
        model = YourEncoderDecoderModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("\nModel loaded successfully from 'model_state_dict'.")
    elif isinstance(checkpoint, dict):
        # Try loading directly as state_dict if no 'model_state_dict' key
        from your_model_file import YourEncoderDecoderModel  # replace this
        model = YourEncoderDecoderModel()
        model.load_state_dict(checkpoint)
        model.eval()
        print("\nModel loaded successfully from checkpoint.")
    else:
        print("\nCheckpoint might already be a full model object.")
        model = checkpoint
        model.eval()
        print("Model loaded directly from checkpoint.")
except Exception as e:
    print("\nCould not load model. You might need the exact model class definition.")
    print("Error:", e)
