

import torch
import torch.nn as nn


class InceptionV3Stem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_a = nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding="valid", bias=False)
        self.bn = nn.BatchNorm2d()
        self.act = nn.ReLU()
        
        self.conv_b = nn.LazyConv2d(out_channels=32, kernel_size=3, stride=1, padding="valid", bias=False)
        self.conv_c = nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding="same", bias=False)
        self.conv_d = nn.LazyConv2d(out_channels=80, kernel_size=1, stride=1, padding="valid", bias=False)
        self.conv_e = nn.LazyConv2d(out_channels=192, kernel_size=3, stride=1, padding="valid", bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
    def forward(self, x):
        # coarse filter of v1 (7x7) factorized into 3x3
        x = self.conv_a(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv_b(x)
        x = self.bn(x)
        x = self.act(x)
        
        # third 3x3. filters are doubled and paddling added
        x = self.conv_c(x)
        x = self.bn(x)
        x = self.act(x)
        
        # pooled feature maps will be reduced by 75%
        x = self.maxpool(x)
        
        # 3x3 reduction
        x = self.conv_d(x)
        x = self.bn(x)
        x = self.act(x)
        
        # Dimensionality expansion
        x = self.conv_e(x)
        x = self.bn(x)
        x = self.act(x)
        
        # pooled feature maps reduce by 75%
        x = self.maxpool(x)
        return x
        
    
        



class InceptionV3Inception(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        
    