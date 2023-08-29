

from torch import nn
import math

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F



import torch
import torch.nn as nn

 
class BasicConv(nn.Module):
 
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x





class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.conv1 = BasicConv(3, 3, kernel_size=3, stride=1, padding=1)
        

        self.conv2 = BasicConv(3, 8, kernel_size=3, stride=1, padding=1)

        self.conv3 = BasicConv(8, 1, kernel_size=3, stride=1, padding=1)


        self.n_length = n_length

    def forward(self, x):

        h, w = x.size(-2), x.size(-1)

        x = x.view((-1, 3) + x.size()[-2:])

        x = self.shallow_conv(x)
        
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))

        for i in range(self.n_length-1):
            x1 = x[:,i,:,:]
            x2 = x[:,i+1,:,:]

            d_i = nn.PairwiseDistance(p=2)(x1, x2).unsqueeze(1)

            d = d_i if i == 0 else torch.cat((d, d_i), 1)

        PA = d.view(-1, 1*(self.n_length-1), h, w)

        return PA
