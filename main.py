import os
import warnings
import argparse
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from glob import glob

from scripts.utils import (
    _get_device,
    prepare_dataset,
    get_normalized_mean,
    get_labelled_indices,
    sabotage_samples,
)
from scripts.trainer import Trainer
from scripts.test import evaluate_test_data
from scripts.model import Unet
from config import Config

warnings.filterwarnings("ignore")


def main(config: object) -> pd.DataFrame:
    """Implement the main function for the pipeline

    Args:
        config (py): Configuration object that contains all the parameters

    Returns:
        pd.DataFrame: Sumamry of all test scores for the model
    """
    # config = Config()
    rng = np.random.RandomState(26)
    seeds = rng.randint(10000, size=config.n_exps)
    mode = config.mode
    RESULTS_DIR = config.RESULTS_DIR
    THRESHOLD = config.THRESHOLD
    device = _get_device()
    os.makedirs(os.path.join(RESULTS_DIR, config.experiment_name), exist_ok=True)
    # Initialize the seeds
    seed = seeds[0]
    # Initialized the object for end to end pipeline
    model = Unet(img_ch=3, output_ch=1, batch_size=config.BATCH_SIZE, device=device).to(
        device
    )
    ssl = Trainer(seed=seed, device=device, model=model, config_file=config)
    # Prepare dataset
    mean_per_channel, std_per_channel = get_normalized_mean(
        sorted(glob(os.path.join((config.train_x), "*")))
    )
    train_dataset, test_dataset = prepare_dataset(
        train_x=sorted(glob(os.path.join((config.train_x), "*")))[:],
        train_y=sorted(glob(os.path.join((config.train_y), "*")))[:],
        valid_x=sorted(glob(os.path.join((config.valid_x), "*")))[:],
        valid_y=sorted(glob(os.path.join((config.valid_y), "*")))[:],
        H=config.H,
        W=config.W,
        mean=mean_per_channel,
        std=std_per_channel,
    )
    labelled_indices = get_labelled_indices(
        train_dataset.images, RATIO_LABELLED_SAMPLES=config.RATIO_LABELLED_SAMPLES
    )
    config.k = len(labelled_indices)
    unlabelled_idxs, train_dataset = sabotage_samples(labelled_indices, train_dataset)
    
    assert np.all(np.array(train_dataset.masks)[unlabelled_idxs[0]] == -1)
    assert np.all(np.array(train_dataset.masks)[labelled_indices[0]] != -1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TEST,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
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
