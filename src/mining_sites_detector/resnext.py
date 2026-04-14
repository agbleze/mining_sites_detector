import torch.nn as nn
import torch


class RestNextStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        
        