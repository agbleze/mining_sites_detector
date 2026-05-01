import torch
import torch.nn as nn
from torchsummary import summary



class XceptionStem(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.conv1 = nn.Sequential(
                                nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(inplace=True)
                            )
        
        self.conv2 = nn.Sequential(nn.LazyConv2d(out_channels=64,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                   nn.LazyBatchNorm2d(),
                                   nn.ReLU(inplace=True)
                                   )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(self)
        
        self.depthwise_conv = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3,
                          stride=1, padding="same",
                          groups=in_channels, bias=False
                          ),
            nn.LazyBatchNorm2d()
            )
        
        self.pointwise_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, 
                                                        kernel_size=1,
                                                        stride=1, padding=0,
                                                        bias=False
                                                        ),
                                            nn.LazyBatchNorm2d()
                                            )
        
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
    
class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 skip_first_relu=False
                 ):
        super().__init__()
        
        self.proj_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
                                                     kernel_size=1,stride=2
                                                     bias=False),
                                            nn.LazyBatchNorm2d()
                                            )
        self.relu = nn.ReLU(inplace=True)
        self.separable_conv1 = DepthwiseSeparableConv(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      )
        self.separable_conv2 = DepthwiseSeparableConv(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        block = []
        if not skip_first_relu:
            block.append(nn.ReLU(inplace=True))
        block.extend([self.separable_conv1, 
                      self.relu, self.separable_conv2, 
                      self.maxpool
                      ]
                     )
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        shortcut = self.proj_conv(x)
        x = self.block(x)
        x += shortcut
        return x
        