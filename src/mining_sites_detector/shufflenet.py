import torch
import torch.nn as nn
from torchinfo import summary
from typing import List, Tuple, Optional, Literal


class ShuffleNetStem(nn.Module):
    def __init__(self, out_channels=24, **kwargs):
        super().__init__()
        
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=2,
                                  bias=False
                                  )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x
    
    