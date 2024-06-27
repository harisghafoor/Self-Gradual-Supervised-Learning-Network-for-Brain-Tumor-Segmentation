import os
import warnings
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from glob import glob

from scripts.utils import savetime, _get_device, prepare_dataset, _get_dataloaders
from scripts.trainer import Trainer
from scripts.test import evaluate_test_data
from scripts.model import Unet
from config import Config

warnings.filterwarnings("ignore")


def main(config):
    rng = np.random.RandomState(26)
    seeds = rng.randint(10000, size=config.n_exps)
    mode = config.mode
    RESULTS_DIR = config.RESULTS_DIR
    THRESHOLD = config.THRESHOLD
    device = _get_device()
    os.makedirs(os.path.join(RESULTS_DIR, config.experiment_name), exist_ok=True)
    # Initialize the seeds
    seed = seeds[0]
    ts = savetime()
    # Initialized the object for end to end pipeline
    model = Unet(img_ch=3, output_ch=1).to(device)
    ssl = Trainer(seed=seed, device=device, model=model, config_file=config)
    # Prepare dataset
    train_dataset, test_dataset = prepare_dataset(
        train_x=sorted(glob(os.path.join((config.train_x), "*")))[:40],
        train_y=sorted(glob(os.path.join((config.train_y), "*")))[:40],
        valid_x=sorted(glob(os.path.join((config.valid_x), "*")))[:40],
        valid_y=sorted(glob(os.path.join((config.valid_y), "*")))[:40],
        H=config.H,
        W=config.W,
    )
    train_loader, _ = _get_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        BATCH_SIZE=config.BATCH_SIZE,
        NUM_WORKERS=config.NUM_WORKERS,
        SHUFFLE_TRAIN=config.SHUFFLE_TRAIN,
        SHUFFLE_TEST=config.SHUFFLE_TEST,
    )

    # Fit the model for each seeds
    if mode == "train":
        model, _ = ssl.fit(train_loader=train_loader, test_dataset=test_dataset)
    # Test the model
    # ssl.device = torch.device("cpu")
    try:
        checkpoint = torch.load(
            ssl.model_save_path,
            map_location=ssl.device,
        )
    except FileNotFoundError:
        print("Model not found")
        return pd.DataFrame()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    df = evaluate_test_data(
        model=model,
        torch_dataset=test_dataset,
        torch_device=ssl.device,
        RESULT_DIR=os.path.join(RESULTS_DIR, config.experiment_name),
        THRESHOLD=THRESHOLD,
        save_csv_file=True,
        save_plots=False,
        show_progress=True,
    )
    return df


if __name__ == "__main__":
    config = Config()
    # mode = 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode parameter (train/test)",
    )
    args = parser.parse_args()

    mode = args.mode
    config.mode = mode
    test_summary = main(config)
    print("test_summary: ", test_summary.mean())
