# %% libraries
import torch.nn as nn
import torch
from models.vgg import vgg16_bn, vgg19_bn
import numpy as np


class CoarseLoss(nn.Module):
    def __init__(self, w1=50, w2=1, weight_vgg=None):
        """
        A weighted sum of pixel-wise L1 loss and sum of L2 loss of Gram matrices.

        :param w1: weight of L1  (pixel-wise)
        :param w2: weight of L2 loss (Gram matrix)
        :param weight_vgg: weight of VGG extracted features (should be add up to 1.0)
        """
        super(CoarseLoss, self).__init__()
        if weight_vgg is None:
            weight_vgg = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.w1 = w1
        self.w2 = w2
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='sum')
        # https://github.com/PatWie/tensorflow-recipes/blob/33962bb45e81f3619bfa6a8aeae5556cc7534caf/EnhanceNet/enet_pat.py#L169

        self.weight_vgg = weight_vgg
        self.vgg16_bn = vgg16_bn(pretrained=True).eval()

    # reference: https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    @staticmethod
    def gram_matrix(mat):
        """
        Return Gram matrix

        :param mat: A matrix  (a=batch size(=1), b=number of feature maps,
        (c,d)=dimensions of a f. map (N=c*d))
        :return: Normalized Gram matrix
        """
        a, b, c, d = mat.size()
        features = mat.view(a * b, c * d)
        gram = torch.mm(features, features.t())
        return gram.div(a * b * c * d)

    def forward(self, y, y_pred):
        y_vgg = self.vgg16_bn(y)
        y_pred_vgg = self.vgg16_bn(y_pred)
        loss_vgg = [self.l2(self.gram_matrix(ly), self.gram_matrix(lp)) for ly, lp in zip(y_vgg, y_pred_vgg)]

        loss = self.w1 * self.l1(y, y_pred) + self.w2 * np.dot(loss_vgg, self.weight_vgg)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        Return Binary Entropy Loss with mean of all losses in each mini-batch
        """
        super(EdgeLoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='mean')

    def forward(self, y, y_pred):
        loss = self.cross_entropy(y, y_pred)
        return loss


class DetailsLoss(nn.Module):
    def __init__(self, w1=100, w2=0.1, w3=0.5, w4=1):
        """

        Return weighted sum of CoarseNet, EdgeNet, DetailsNet and Adversarial losses averaged over
        all losses in each mini-batch.

        :param w1: Weight of CoarseNet loss
        :param w2: Weight of EdgeNet loss
        :param w3: Weight of Local Patch loss
        :param w4: Weight of Adversarial loss
        """

        super(DetailsLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.MSE_loss = nn.MSELoss(reduction='mean')
        self.BCE_loss = nn.BCELoss(reduction='mean')
        self.vgg19_bn = vgg19_bn(pretrained=True).eval()

    # reference: https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    @staticmethod
    def gram_matrix(mat):
        """
        Return Gram matrix

        :param mat: A matrix  (a=batch size(=1), b=number of feature maps, p=number of patches,
        (c,d)=dimensions of a f. map (N=c*d))

        :return: Normalized Gram matrix
        """
        a, b, p, c, d = mat.size()
        features = mat.view(a * b * p, c * d)
        gram = torch.mm(features, features.t())
        return gram.div(a * b * p * c * d)

    @staticmethod
    def get_patch(mat, size=14, stride=14):
        """
        Returns a tensor of patches of input tensor

        :param mat: A tensor (batch_size, channel_size, height, width)
        :param stride: Stride size of the patch
        :param size: Size of the patch
        :return: A patched tensor (batch_size, channel_size, patch_size, height, width)
        """

        batch_size, channel_size, height, width = tuple(mat.size())
        patches = mat.unfold(2, size, stride).unfold(3, size, stride)
        patches = patches.contiguous().view((batch_size, channel_size, -1, size, stride))
        return patches

    def forward(self, y, y_pred):
        """

        :param y: Ground truth tensor
        :param y_pred: Estimated ground truth
        :return: A scalar number
        """

        # TODO y_pred and y are concatenated latent vector, so first we must extract different features.

        y_vgg = self.vgg16_bn(y)
        y_pred_vgg = self.vgg16_bn(y_pred)
        coarse_loss = self.l1_loss(y, y_pred)
        edge_loss = self.BCE_loss(y, y_pred)
        patch_loss = np.sum(
            [self.MSE_loss(self.gram_matrix(self.get_patch(ly)), self.gram_matrix(self.get_patch(lp)))
             for ly, lp in zip(y_vgg, y_pred_vgg)])
        adversarial_loss = self.MSE_loss(y, y_pred)

        loss = self.w1 * coarse_loss + self.w2 * edge_loss + self.w3 * patch_loss + self.w4 * adversarial_loss
        return loss
