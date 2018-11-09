# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:56:10 2018

@author: Mohammad Doosti Lakhani
"""


# Importing Libraries
import torch
from torch import nn
import torch.nn.functional as f


class U_Net(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, depth=5, number_of_filters=64):
        """
        Implementation of U-Net.
        Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        [https://arxiv.org/abs/1505.04597]
        
        Note: Default arguments are based on mentioned paper implementation.
        
        Args:
            **input_channels**: number of input channels of input images to network.
            
            **output_channels**: number of output channels of output images of network.
            
            **depth**: depth of network
            
            **number_of_filters**: number of filters in each layer (Each layer x2 the value).
            
        """
        super(U_Net, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.number_of_filters = number_of_filters
        
        self.contracting_path = nn.ModuleList() # left side of shape of paper
        self.expansive_path = nn.ModuleList() # right side of shape of paper
    
    
    
        new_out_channels = self.number_of_filters
        new_in_channels = self.input_channels
        
        # Filling pathes with correspoding layers.
        for i in range(depth):
            self.contracting_path.append(DownConvOlution(new_in_channels, new_out_channels))
            new_in_channels = new_out_channels
            new_out_channels = new_out_channels*2
        
        new_out_channels= new_out_channels // 2
        
        for i in reversed(range(depth-1)):
            new_out_channels = new_out_channels // 2
            self.expansive_path.append(UpConvolution(new_in_channels, new_out_channels))
            new_in_channels = new_out_channels
            
        self.last_layer = nn.Conv2d(new_in_channels, output_channels, kernel_size=1)   
        
        
            