# %% import library
import torch
import torch.nn as nn
from models.layers import CL, CBL, C


# %% discriminator one
class DiscriminatorOne(nn.Module):
    def __init__(self, input_channel=3, output_channel=1):
        """
        Consists of a CL module followed by repetitive CBL modules and finally a C class
        to match the final needed classes.

        :param input_channel: number of input channels of input images to network.
        :param output_channel: number of output channels of input images to network.
        """

        super(DiscriminatorOne, self).__init__()
        self.cl = CL(input_channel=input_channel, output_channel=128, kernel_size=4, stride=2, padding=1)
        self.cbl0 = CBL(input_channel=128, output_channel=256, kernel_size=4, stride=2, padding=1)
        self.cbl1 = CBL(input_channel=256, output_channel=512, kernel_size=4, stride=2, padding=1)
        self.cbl2 = CBL(input_channel=512, output_channel=1024, kernel_size=4, stride=2, padding=1)
        self.cbl3 = CBL(input_channel=1024, output_channel=2048, kernel_size=4, stride=2, padding=1)

        self.final = C(input_channel=2048, output_channel=output_channel, kernel_size=1, stride=1, padding=0,
                       activation=None)

    def forward(self, x):
        x = self.cl(x)
        x = self.cbl0(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        x = self.final(x)
        return x


# %% discriminator two
class DiscriminatorTwo(nn.Module):
    def __init__(self, input_channel=9, output_channel=1):
        """
        Consists of a CL module followed by repetitive CBL modules and finally a C class
        to match the final needed classes.

        :param input_channel: number of input channels of input images to network which is concatenation of
        I<sub>h</sub>, I<sub>d</sub>, and I<sub>o</sub> RGB vectors.
        :param output_channel: number of output channels of input images to network.
        """

        super(DiscriminatorTwo, self).__init__()
        self.cl = CL(input_channel=input_channel, output_channel=128, kernel_size=5, stride=2, padding=0)
        self.cbl0 = CBL(input_channel=128, output_channel=256, kernel_size=5, stride=2, padding=0)
        self.cbl1 = CBL(input_channel=256, output_channel=512, kernel_size=5, stride=2, padding=0)
        self.cbl2 = CBL(input_channel=512, output_channel=1024, kernel_size=5, stride=2, padding=0)
        self.cbl3 = CBL(input_channel=1024, output_channel=2048, kernel_size=5, stride=2, padding=0)

        self.final = C(input_channel=2048, output_channel=output_channel, kernel_size=4, stride=1, padding=0,
                       activation='None')

    def forward(self, x):
        x = self.cl(x)
        x = self.cbl0(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        x = self.final(x)
        return x


# %% tests
# z = torch.randn(size=(1, 3, 256, 256))
# d1 = DiscriminatorOne()
# z = d1(z)
# z.size()
