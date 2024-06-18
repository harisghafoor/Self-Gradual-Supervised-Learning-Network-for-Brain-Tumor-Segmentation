import numpy as np
from timeit import default_timer as timer
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    
    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05,device=None):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape)).to(device)
        self.std = std
        
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise
    
class CNN(nn.Module):
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32,device=None):
        super(CNN, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = GaussianNoise(batch_size, std=self.std,device=device)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10,device=device)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x