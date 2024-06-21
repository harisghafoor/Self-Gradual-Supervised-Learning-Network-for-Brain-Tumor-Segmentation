import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from utils import savetime, save_exp
from trainer import Trainer
from config import Config

rng = np.random.RandomState(26)
# Initialize Config File
config = Config()
seeds = rng.randint(10000, size=config.n_exps)
# Initialize the seeds
seed = seeds[0]
ts = savetime()
# Initialized the object for end to end pipeline
ssl = Trainer(seed=seed) 
# Fit the model for each seeds
ssl.fit()
# Store the losses in containers
