import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DCBlock(nn.Module):
    """Dilated CNN-stack (DC) block with dense connections"""
    
    def __init__(self, in_channels, out_channels, dilation_rate=1, kernel_size=3, groups=1):
        super(DCBlock, self).__init__()
        
        padding = dilation_rate * (kernel_size - 1) // 2
        
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation_rate, padding=padding, groups=groups, bias=False
        )
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return torch.cat([out, x], dim=1)


