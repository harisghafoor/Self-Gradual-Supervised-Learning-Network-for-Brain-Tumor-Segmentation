import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import weight_norm


class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05, device=None):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape)).to(device)
        self.std = std

    def forward(self, x):
        # print("Noise Shape : ",self.noise.shape)
        # print("X Shape : ",x.shape)
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise


# class CNN(nn.Module):

#     def __init__(
#         self,
#         batch_size,
#         std,
#         p=0.5,
#         fm1=16,
#         channel_input=1,
#         channgel_output=10,
#         device=None,
#     ):
#         super(CNN, self).__init__()
#         self.device = device
#         self.input_num_channels = channel_input
#         self.output_num_channels = channgel_output
#         self.fm1 = fm1
#         self.std = std
#         self.gn = GaussianNoise(
#             batch_size,
#             std=self.std,
#             input_shape=(channel_input, 28, 28),
#             device=self.device,
#         )
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(p)
#         self.conv1 = nn.Conv2d(channel_input, self.fm1, 3, padding=1)
#         self.conv2 = nn.Conv2d(self.fm1, 2 * self.fm1, 3, padding=1)
#         self.conv3 = nn.Conv2d(2 * self.fm1, 4 * self.fm1, 3, padding=1)
#         self.conv4 = nn.Conv2d(4 * self.fm1, 6 * self.fm1, 3, padding=1)
#         self.conv5 = nn.Conv2d(6 * self.fm1, 8 * self.fm1, 3, padding=1)
#         self.conv6 = nn.Conv2d(8 * self.fm1, 16 * self.fm1, 3, padding=1)
#         # self.conv1 = weight_norm(nn.Conv2d(channel_input, self.fm1, 3, padding=1))
#         # self.conv2 = weight_norm(nn.Conv2d(self.fm1, 2 * self.fm1, 3, padding=1))
#         # self.conv3 = weight_norm(nn.Conv2d(2 * self.fm1, 4 * self.fm1, 3, padding=1))
#         # self.conv4 = weight_norm(nn.Conv2d(4 * self.fm1, 6 * self.fm1, 3, padding=1))
#         # self.conv5 = weight_norm(nn.Conv2d(6 * self.fm1, 8 * self.fm1, 3, padding=1))
#         # self.conv6 = weight_norm(nn.Conv2d(8 * self.fm1, 16 * self.fm1, 3, padding=1))
#         self.mp = nn.MaxPool2d(3, stride=2, padding=1)

#     def forward(self, x):
#         if self.training:
#             x = self.gn(x)

#         x1 = self.conv1(x)
#         x1 = self.mp(x1)
#         x1 = self.act(x1)

#         x2 = self.conv2(x1)
#         x2 = self.mp(x2)
#         x2 = self.act(x2)

#         # x3 = self.conv3(x2)
#         # x3 = self.mp(x3)
#         # x3 = self.act(x3)

#         # x4 = self.conv4(x3)
#         # x4 = self.mp(x4)
#         # x4 = self.act(x4)

#         # x5 = self.conv5(x4)
#         # x5 = self.mp(x5)
#         # x5 = self.act(x5)

#         # x6 = self.conv6(x5)
#         # x6 = self.mp(x6)
#         # x6 = self.act(x6)

#         # print("shape of x4: ", x2.shape)
#         # x = self.act(self.mp(self.conv1(x)))
#         # x = self.act(self.mp(self.conv2(x)))
#         # x = x.view(-1, 16 * self.fm1 * 7 * 7)
#         # print("x2 shape: ", x2.shape)
#         x7 = x2.view(-1, x2.shape[1] * x2.shape[2] * x2.shape[3])
#         # x7 = x6.view(-1, x6.shape[1] * x6.shape[2] * x6.shape[3])
#         dropped_out = self.drop(x7)
#         # print("x3 shape: ", x3.shape)
#         # print("Shape for Fully Connected Layer: ", x2.shape)
#         fully_connected_layer = nn.Linear(
#             x2.shape[1] * x2.shape[2] * x2.shape[3],
#             self.output_num_channels,
#             device=self.device,
#         )
#         output = fully_connected_layer(dropped_out)
#         # print("shape of output: ", output.shape)
#         # x = self.fc(x)
#         return output
#         # return dropped_out


# class GaussianNoise(nn.Module):

#     def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05,device=None):
#         super(GaussianNoise, self).__init__()
#         self.shape = (batch_size,) + input_shape
#         self.noise = Variable(torch.zeros(self.shape)).to(device)
#         self.std = std

#     def forward(self, x):
#         self.noise.data.normal_(0, std=self.std)
#         return x + self.noise


class CNN(nn.Module):
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32, device=None):
        super(CNN, self).__init__()
        self.fm1 = fm1
        self.fm2 = fm2
        self.std = std
        self.device = device
        self.gn = GaussianNoise(batch_size, std=self.std, device=device)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, 2 * self.fm1, 3, padding=1))
        self.conv3 = weight_norm(nn.Conv2d(2 * self.fm1, self.fm1 * 4, 3, padding=1))
        self.conv4 = weight_norm(nn.Conv2d(self.fm1 * 4, self.fm1 * 8, 3, padding=1))
        self.conv5 = weight_norm(nn.Conv2d(self.fm1 * 8, self.fm1 * 16, 3, padding=1))
        self.conv6 = weight_norm(nn.Conv2d(self.fm1 * 16, self.fm1 * 32, 3, padding=1))
        self.mp = nn.MaxPool2d(3, stride=4, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 10, device=device)

    def forward(self, x):
        if self.training:
            x = self.gn(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mp(x)
        x = self.act(x)

        # x = self.conv2(x)
        # x = self.mp(x)
        # x = self.act(x)

        # x = self.conv3(x)
        # x = self.mp(x)
        # x = self.act(x)

        # x = self.conv4(x)
        # # x = self.mp(x)
        # x = self.act(x)

        # x = self.conv5(x)
        # x = self.mp(x)
        # x = self.act(x)

        # x = self.conv6(x)
        # x = self.mp(x)
        # x = self.act(x)

        # x = self.act(self.mp(self.conv1(x)))
        # # print("shape of x1: ", x.shape)
        # x = self.act(self.mp(self.conv2(x)))
        # # print("shape of x2: ", x.shape)
        # x = self.act(self.mp(self.conv3(x)))
        # print("shape of x3: ", x.shape)
        x_prime = x
        # print("shape of x: ", x.shape)
        x = x.view(-1, x_prime.shape[1] * x_prime.shape[2] * x_prime.shape[3])
        # print("shape of x: ", x.shape)
        x = self.drop(x)
        x = self.fc(x)
        # print("At the momemnt the shape of x is: ", x_prime.shape)
        # x = nn.Linear(
        #     x_prime.shape[1] * x_prime.shape[2] * x_prime.shape[3],
        #     10,
        #     device=self.device,
        # )(x)

        # x = nn.Linear(
        #     16 * 14 * 14,
        #     10,
        #     device=self.device,
        # )(x)

        return x


if __name__ == "__main__":
    device = torch.device("mps")
    model = CNN(100, 0.05, device=device).to(device)
    # model.train()
    # print("Printing the Model Behaviour During training",model.training)
    # model.eval()
    # print("Printing the Model Behaviour During evaluation",model.training)
    input_x = torch.randn(100, 1, 28, 28).to(device)
    output_x = model(input_x)
    print(input_x.shape)
    print(output_x.shape)
