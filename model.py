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


""" Multi Output Attention Unet"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        # self.conv_transposed = nn.Upsample(scale_factor=2)
        self.up_medium = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=1
        )
        # self.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print("Upsampling Shape : ", self.conv_transposed(x).shape)
        # print("Upsampling Shape By medium method: ", self.up_medium(x).shape)
        x = self.up(x)
        return x


class Unet(nn.Module):
    """
    AttentionUNetppGradual is a class that implements the Attention UNet++ model with gradual supervision for superior segmentation.

    Args:
        img_ch (int): Number of input channels (default: 3)
        output_ch (int): Number of output channels (default: 1)
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(Unet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)

        # Bottleneck convolutional operation\
        # self.Conv5 = ConvBlock(128, 256)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up4 = UpConv(1024, 512)
        self.UpConv4 = ConvBlock(1024, 512)

        self.Up3 = UpConv(512, 256)
        self.UpConv3 = ConvBlock(512, 256)

        self.Up2 = UpConv(256, 128)
        self.UpConv2 = ConvBlock(256, 128)

        self.Up1 = UpConv(128, 64)
        self.UpConv1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the AttentionUNetppGradual model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, output_channels, height, width)
            ds_out (list): List of deep supervision outputs, each of shape (batch_size, output_channels, height, width)
        """
        e1 = self.Conv1(x)
        # print("Conv1 Shape : ", e1.shape)
        p1 = self.MaxPool(e1)
        # print("Max Pool 1 Shape : ", p1.shape)

        e2 = self.Conv2(p1)
        # print("Conv2 Shape : ", e2.shape)
        p2 = self.MaxPool(e2)
        # print("Max Pool 2 Shape : ", p2.shape)

        e3 = self.Conv3(p2)
        # # print("Conv3 Shape : ", e3.shape)
        p3 = self.MaxPool(e3)
        # # print("MAx Pool 3 Shape : ", p3.shape)

        e4 = self.Conv4(p3)
        # # print("Conv4 Shape : ", e4.shape)
        p4 = self.MaxPool(e4)
        # # print("Max pool4 Shape : ", p4.shape)

        bottle_neck = self.Conv5(p4)
        # bottle_neck = self.Conv5(p2)

        # print("Bottleneck Shape : ", bottle_neck.shape)

        d4 = self.Up4(bottle_neck)
        # # print("Deconv4 Shape : ", d4.shape)
        d4 = torch.cat((d4, e4), dim=1)
        # # print("Skip Connection 4 Shape : ", d4.shape)
        d4 = self.UpConv4(d4)
        # # print("Deconv4 Final Shape : ", d4.shape)

        d3 = self.Up3(d4)
        # # print("D7 Shape : ", d3.shape)
        d3 = torch.cat((d3, e3), dim=1)
        # # print("Skip Connection Deconv-3 Shape : ", d3.shape)
        d3 = self.UpConv3(d3)
        # # print("Deconv3 Final Shape : ", d3.shape)

        d2 = self.Up2(d3)
        # d2 = self.Up2(bottle_neck)
        # print("Deconv2 Shape : ", d2.shape)
        d2 = torch.cat((d2, e2), dim=1)
        # print("Skip Connection Deconv-2 Shape : ", d2.shape)
        d2 = self.UpConv2(d2)
        # print("Deconv2 Final Shape : ", d2.shape)


        d1 = self.Up1(d2)
        # print("Deconv2 Shape : ", d1.shape)
        d1 = torch.cat((d1, e1), dim=1)
        # print("Skip Connection Deconv-1 Shape : ", d1.shape)
        d1 = self.UpConv1(d1)
        # print("Deconv1 Final Shape : ", d1.shape)

        out = self.final_conv(d1)
        # out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    device = torch.device("cpu")
    # x = torch.randn((16, 3, 512, 512)).to(device)
    f = Unet().to(device)
    # main_output = f(x)
    # print("Main Input Shape:", x.shape)
    # print("Main Output Shape:", main_output.shape)
    # print(f)
    print(summary(f, (3, 512, 512)))
#     print("Second Last Decoder Output Shape:", ds_outputs[0].shape)
#     print("Third Last Decoder Output Shape:", ds_outputs[1].shape)
