import torch
import torch.nn as nn


class CL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a leaky rectified linear unit (Leaky ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """

        super(CL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.LeakyReLU(0.2, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CBL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, and a batch normalization, followed by
        a leaky rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(num_features=output_channel), nn.LeakyReLU(0.2, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CBR(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        """
        It consists of the 5x5 convolutions with stride=1, padding=2, and a batch normalization, followed by
        a rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBR, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(num_features=output_channel), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CE(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a exponential linear unit (ELU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        :param kernel_size: kernel size
        :param stride: stride size
        """

        super(CE, self).__init__()
        layers = [nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.ELU(alpha=1, inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class C(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, activation=None):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(C, self).__init__()
        if activation == 'sigmoid':
            self.layer = nn.Sequential(
                [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                 nn.Sigmoid()])
        elif activation == 'tanh':
            self.layer = nn.Sequential(
                [nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                 nn.Tanh()])
        else:
            self.layer = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.layer(x)

