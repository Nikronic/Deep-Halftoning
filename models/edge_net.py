import torch
import torch.nn as nn
from models.layers import CBR, C


class EdgeNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        """
        A simple convolutional neural network to learn edges using Canny edge detector

        :param input_channels: number of input channels of input images to network.
        :param output_channels: number of output channels of output images of network.
        """

        super(EdgeNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.cbr0 = CBR(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.cbr1 = CBR(32, 32, kernel_size=5, stride=1, padding=2)
        self.cbr2 = CBR(32, 32, kernel_size=5, stride=1, padding=2)
        self.cbr3 = CBR(32, 32, kernel_size=5, stride=1, padding=2)
        self.cbr4 = CBR(32, 32, kernel_size=5, stride=1, padding=2)

        # final
        self.final = C(32, self.output_channels, kernel_size=3, stride=1, padding=1, activation='sigmoid')

    def forward(self, x):
        c = self.cbr0(x)
        c = self.cbr1(c)
        c = self.cbr2(c)
        c = self.cbr3(c)
        c = self.cbr4(c)

        c = self.final(c)
        return c

