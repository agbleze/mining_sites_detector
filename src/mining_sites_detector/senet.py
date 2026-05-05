import torch
import torch.nn as nn



class SenetStem(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.conv1 = nn.LazyConv2d(out_channels=64, 
                                   kernel_size=7,
                                   stride=2
                                   )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        return x