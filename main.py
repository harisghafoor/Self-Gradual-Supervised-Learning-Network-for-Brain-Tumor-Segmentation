import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from utils import savetime, save_exp
from trainer import Trainer
from config import Config

# metrics
accs         = []
accs_best    = []
losses       = []
sup_losses   = []
unsup_losses = []
idxs         = []

# Initialize Config File
config = Config()
ts = savetime()

# Loop through each experiment
for i in range(config.n_exp):
    # Initialized the object for end to end pipeline
    ssl = Trainer(seed=config.seeds[i]) 
    # Fit the model for each seeds
    acc, acc_best, l, sl, usl, indices = ssl.fit()
    # Store the losses in containers
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)
    
    
    
# Save all stats
print ('saving experiment')

try:
    save_exp(ts, losses, sup_losses, unsup_losses,
             accs, accs_best, idxs, config=config)
except Exception as e:
    print(f"Error in the storing of values as {e}")
