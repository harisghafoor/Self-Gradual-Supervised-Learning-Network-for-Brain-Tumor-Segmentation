import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from utils import savetime, save_exp
from trainer import Trainer
from config import Config
import torch.nn.functional as F
from eval import compute_loss, DiceLoss, calculate_metrics
from test import evaluate_test_data
import os
import warnings
import argparse
warnings.filterwarnings("ignore")

def main(config):
    rng = np.random.RandomState(26)
    seeds = rng.randint(10000, size=config.n_exps)
    mode = config.mode
    RESULTS_DIR = config.RESULTS_DIR
    THRESHOLD = config.THRESHOLD
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Initialize the seeds
    seed = seeds[0]
    ts = savetime()
    # Initialized the object for end to end pipeline
    ssl = Trainer(seed=seed) 
    # Fit the model for each seeds
    if mode == 'train':
        model, losses = ssl.fit()
    # Test the model   
    # ssl.device = torch.device("cpu")
    model = ssl.model.to(ssl.device)
    checkpoint = torch.load(
        "./model_best.pth.tar",
        map_location=ssl.device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    df = evaluate_test_data(model=model,
                            torch_dataset = ssl.test_dataset,
                            torch_device = ssl.device,
                            RESULT_DIR = RESULTS_DIR,
                            THRESHOLD = THRESHOLD)
    return df
    
if __name__ == "__main__":
    config = Config()
    # mode = 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode parameter (train/test)")
    args = parser.parse_args()

    mode = args.mode
    config.mode = mode
    test_summary = main(config)
    print("test_summary: ", test_summary.mean())