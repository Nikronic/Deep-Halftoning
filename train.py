# %% import libraries
# ours
from models.coarse_net import CoarseNet
from models.edge_net import EdgeNet
from models.details_net import DetailsNet
from models.discriminators import DiscriminatorOne, DiscriminatorTwo
from utils.losses import CoarseLoss, EdgeLoss, DetailsLoss
from utils.preprocess import *

# Pytorch
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip, Normalize
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

# ObjectNet requirements
# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs

from models.object_net import ModelBuilder, SegmentationModule
from utils.object_net_utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm


# %% global variable initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %% weight initializer
def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"

    :param m: Layer to initialize
    :return: None
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  # reference: https://github.com/pytorch/pytorch/issues/12259
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# %% simulating argparse module
class args:
    txt = 'dataset/sub_test/filelist.txt'
    img = 'dataset/sub_test/data'
    txt_t = 'dataset/sub_test/filelist.txt'
    img_t = 'dataset/sub_test/data'
    bs = 128
    nw = 4
    es = 20
    lr = 0.0001
    cudnn = 0
    pm = 0

# TODO to determine number of epoch size, we have to consider the concept of augmentation in pytorch
# https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch/54460259#54460259


if args.cudnn == 1:
    cudnn.benchmark = True
else:
    cudnn.benchmark = False

if args.pm == 1:
    pin_memory = True
else:
    pin_memory = False

# %% define datasets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    RandomNoise(p=0.5, mean=0, std=0.1)])

train_dataset = PlacesDataset(txt_path=args.txt,
                              img_dir=args.img,
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.bs,
                          shuffle=True,
                          num_workers=args.nw,
                          pin_memory=pin_memory)

test_dataset = PlacesDataset(txt_path=args.txt_t,
                             img_dir=args.img_t,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=args.nw,
                         pin_memory=False)


# %% train model
def train_model(network, data_loader, optimizer, criterion, epochs=2):
    """
    Train model

    :param networks: Parameters of defined neural networks
    :param data_loader: A data loader object defined on train data set
    :param epochs: Number of epochs to train model
    :param optimizer: Optimizer to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            X = data['X']
            y_d = data['y_descreen']

            X = X.to(device)
            y_d = y_d.to(device)

            optimizer.zero_grad()

            outputs = net(X)
            loss = criterion(outputs, y_d)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(epoch + 1, ',', i + 1, 'loss:', running_loss)
    print('Finished Training')


# %% test
def test_model(net, data_loader):
    """
    Return loss on test

    :param net: The trained NN network
    :param data_loader: Data loader containing test set
    :return: Print loss value over test set in console
    """

    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            y_descreen = data['y_descreen']
            y_e = data['y_edge']

            y_descreen = y_descreen.to(device)
            y_e = y_e.to(device)
            outputs = net(y_descreen)
            loss = criterion(outputs, y_e)
            running_loss += loss
            print('loss: %.3f' % running_loss)
    return outputs


def show_batch_image(image_batch):
    """
    Show a sample grid image which contains some sample of test set result

    :param image_batch: The output batch of test set
    :return: PIL image of all images of the input batch
    """

    to_pil = ToPILImage()
    fs = []
    for i in range(len(image_batch)):
        img = to_pil(image_batch[i].cpu())
        fs.append(img)
    x, y = fs[0].size
    ncol = int(np.ceil(np.sqrt(len(image_batch))))
    nrow = int(np.ceil(np.sqrt(len(image_batch))))
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * int(i / nrow), y * (i % nrow)
        cvs.paste((fs[i]), (px, py))
    cvs.save('out.png', format='png')
    cvs.show()


# %% initialize network, loss and optimizer

# CoarseNet
coarse_loss = CoarseLoss(w1=50, w2=1).to(device)
coarse_net = CoarseNet().to(device)
coarse_optim = optim.Adam(coarse_net.parameters(), lr=args.lr)
coarse_net.apply(init_weights)

# EdgeNet
edge_loss = EdgeLoss().to(device)
edge_net = EdgeNet().to(device)
edge_optim = optim.Adam(edge_net.parameters(), lr=args.lr)
edge_net.apply(init_weights)

# ObjectNet
builder = ModelBuilder()
net_encoder = builder.build_encoder(
    arch='resnet101dilated',
    fc_dim=2048,
    weights=os.path.join('pretrained/baseline-resnet101dilated-ppm_deepsup', 'encoder' + '_epoch_25.pth'))
net_decoder = builder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights=os.path.join('pretrained/baseline-resnet101dilated-ppm_deepsup', 'decoder' + '_epoch_25.pth'),
    use_softmax=True)
segmentation_module = SegmentationModule(net_encoder, net_decoder, None)
segmentation_module.cuda()

# DetailsNet
details_loss = DetailsLoss().to(device)
random_noise_adder = RandomNoise(p=0, mean=0, std=1)  # add noise to input of generator (DetailsNet)
details_net = DetailsNet().to(device)
disc_one = DiscriminatorOne().to(device)
disc_two = DiscriminatorTwo().to(device)

details_optim = optim.Adam(details_net.parameters(), lr=args.lr)
disc_one_optim = optim.Adam(disc_one.parameters(), lr=args.lr)
disc_two_optim = optim.Adam(disc_two.parameters(), lr=args.lr)

details_net.apply(init_weights)
disc_one.apply(init_weights)
disc_two.apply(init_weights)


# %% Train model

models = {
    'coarse': coarse_net,
    'edge': edge_net,
    'object': segmentation_module,
    'details': details_net,
    'disc1': disc_one,
    'disc2': disc_two
}

losses = {
    'coarse': coarse_loss,
    'edge': edge_loss,
    'details': details_loss
}

optims = {
    'coarse': coarse_optim,
    'edge': edge_optim,
    'details': details_optim,
    'disc1': disc_one_optim,
    'disc2': disc_two_optim
}


train_model(network=models, data_loader=train_loader, optimizer=optims, criterion=losses, epochs=args.es)


