

import torch
import torch.nn as nn
import numpy as np




class inceptionV2Block(nn.Module):
    def __init__(self, f1xf1, f3xf3, f5x5, fpool):
        super().__init__()
        
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d()
        
        self.zeropad = nn.ZeroPad2d()
        self.bpool = nn.MaxPool2d(kernel_size=3)
        self.bpool_conv1x1 = nn.Conv2d(out_channels=fpool[0], kernel_size=1, stride=1, padding="same")
        
        self.f1xf1_conv1x1 = nn.LazyConv2d(out_channel=f1xf1[0], kernel_size=1, stride=1, padding="same")
        
        self.f3x3_conv1x1_b = nn.LazyConv2d(out_channels=f3xf3[0], kernel_size=1, stride=1, padding="same")
        self.f3x3_conv3x3 = nn.LazyConv2d(out_channels=f3xf3[1], kernel_size=3, stride=1, padding="valid")
        
        self.f5x5_conv1x1 = nn.LazyConv2d(out_channels=f5x5[0], kernel_size=1, stride=1, padding="same")
        self.f5x5_conv3x3_a = nn.LazyConv2d(out_channels=f5x5[1], kernel_size=3, stride=1, padding="valid")
        self.f5x5_conv3x3_b = nn.LazyConv2d(out_channels=f5x5[2], kernel_size=3, stride=1, padding="valid")
        
        
    def forward(self, x):
        x_f1x1 = self.f1xf1_conv1x1(x)
        x_f1x1 = self.bn(x_f1x1)
        x_f1x1 = self.act(x_f1x1)
        
        
        x_f3x3 = self.f3x3_conv1x1_b(x)
        x_f3x3 = self.bn(x_f3x3)
        x_f3x3 = self.act(x_f3x3)
        x_f3x3 = self.zeropad(x_f3x3)
        
        x_f3x3 = self.f3x3_conv3x3(x_f3x3)
        x_f3x3 = self.bn(x_f3x3)
        x_f3x3 = self.act(x_f3x3)
        
        
        x_f5x5 = self.f5x5_conv1x1(x)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        x_f5x5 = self.zeropad(x_f5x5)
        x_f5x5 = self.f5x5_conv3x3_a(x_f5x5)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        x_f5x5 = self.f5x5_conv3x3_b(x_f5x5)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        
        x_bpool = self.bpool(x)
        x_bpool = self.bpool_conv1x1(x_bpool)
        x_bpool = self.bn(x_bpool)
        x_bpool = self.act(x_bpool)

        output = torch.concat([x_f1x1, x_f3x3, x_f5x5, x_bpool])
        return output




