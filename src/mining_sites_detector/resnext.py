import torch.nn as nn
import torch


class ResNextStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d()
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)
        return x
        
        
        
        