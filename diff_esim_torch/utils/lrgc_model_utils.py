# custom cnn layer for our learned spatial temporal kernel 

import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import sys
import math
from .lrgc_emulator_utils import initialize_identity


class lstkConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, bias, padding, padding_mode, custom_kernels=None, dtype=None, mode='cur', groups=1):
        super(lstkConv, self).__init__()
        assert mode in ['cur', 'mem']
        self.event_conv = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, groups=groups,  padding_mode=padding_mode, dtype=dtype)
        self.out_channels = out_channels
        initialize_identity(self.event_conv.weight, kernel_size, dtype=self.event_conv.weight.dtype, mode=mode)

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.dtype)    
        # print(self.event_conv.weight.dtype)
        # sys.exit()
        out = self.event_conv(x)
        # print("lstk conv out ",out.size())
        # tiled = torch.zeros_like(x, device=x.device, dtype=out.dtype)
        tiled = torch.zeros([b,self.out_channels, h, w], device=x.device, dtype=out.dtype)
        # print(tiled.size())
        # sys.exit()

        tiled[:,:,0::2, 0::2] = out[:,0:1,0::2, 0::2]
        tiled[:,:,0::2, 1::2] = out[:,1:2,0::2, 1::2]
        tiled[:,:,1::2, 0::2] = out[:,2:3,1::2, 0::2]
        tiled[:,:,1::2, 1::2] = out[:,3:4,1::2, 1::2]
        # print(tiled.size())
        # sys.exit()
        return tiled
