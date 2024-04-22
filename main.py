import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

from trainer import trainer
from model import convnet
from dataset import mnist
from config import Config


def main():
    cnn_model = trainer(model=convnet(batch_size=100,
                                  std=0.5),
                    seed=Config.seeds[0],dataset = mnist,configs = Config())
    train_loader,test_loader,idx = cnn_model.prepare_dataloader()
    cnn_model.fit(train_loader= train_loader,
              test_loader=test_loader,
              )
if __name__ == '__main__':
    main()
