from models.encoder import SFCEncoder
from models.decoder import VNFDecoder
from models.gnn_heuristic import GNNHeuristic
import torch

encoder = SFCEncoder()
decoder = VNFDecoder()
gnn = GNNHeuristic()

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))
gnn.load_state_dict(torch.load("gnn_heuristic.pth"))

print("All models loaded successfully.")
