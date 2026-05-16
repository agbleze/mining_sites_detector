import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List



class MobileNetV3Stem(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        self.conv = nn.LazyConv2d(kernel_size=3, out_channels=out_channels,
                                  bias=False,
                                  stride=2
                                  )
        self.hswish = nn.Hardswish()
        self.bn = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        return x
    
    
        
        