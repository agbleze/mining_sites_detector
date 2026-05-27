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
    
    
    
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        
        # Reshape the input tensor to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose the groups and channels_per_group dimensions
        x = x.transpose(1, 2).contiguous()
        
        # Reshape back to the original shape
        x = x.view(batch_size, num_channels, height, width)
        
        return x