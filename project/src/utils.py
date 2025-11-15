import argparse
import os
import torch
import numpy as np
from datetime import datetime

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model_dict(model, path, map_location='cpu'):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_ants', type=int, default=32)
    parser.add_argument('--num_colonies', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='experiments')
    return parser.parse_args()

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
