import torch
import torchvision.datasets as datasets

class mnist(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.n = len(data)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]