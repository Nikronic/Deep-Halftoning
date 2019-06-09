import torch
import torch.nn as nn
from models.layers import CBL, CL, C


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        """
        A residual block contains a sequence of CBL and CL classes with same size of stride, padding and kernel size.

        :param input_channel: number of input channels of input images to network.
        :param output_channel: number of output channels of output images of network.
        :param stride: stride size of CBL and CL modules
        :param kernel_size: kernel_size of CBL and CL modules
        :param padding: padding size of CBL and CL modules
        """

        super(ResidualBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        # blocks
        self.cbl = CBL(input_channel, output_channel, kernel_size, stride, padding)
        self.cl = CL(output_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        out = self.cbl(x)
        out = self.cl(out)
        return out


class DetailsNet(nn.Module):
    def __init__(self, input_channels=32, output_channels=3):
        """
        The generator of GAN networks contains repeated residual blocks and C block at the end.

        :param input_channels: number of input channels of input images to network. Actually, it is latent vector length
        which is fusion of I<sub>h</sub>, I<sub>a</sub>, I<sub>c</sub>, I<sub>e</sub> vectors
        which is called I<sub>f</sub>.
        :param output_channels: number of output channels of output images of network.
        """

        super(DetailsNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block0 = ResidualBlock(input_channel=self.input_channels, output_channel=64)
        self.block1 = ResidualBlock(input_channel=64, output_channel=64)
        self.block2 = ResidualBlock(input_channel=64, output_channel=64)
        self.block3 = ResidualBlock(input_channel=64, output_channel=64)

        self.final = C(input_channel=64, output_channel=self.output_channels,
                       kernel_size=3, stride=1, padding=1, activation='tanh')

    def forward(self, x):
        x = self.block0(x)

        residual1 = x
        x = self.block1(x)
        x += residual1

        residual2 = x
        x = self.block3(x)
        x += residual2

        residual3 = x
        x = self.block3(x)
        x += residual3

        x = self.final(x)

        return x

