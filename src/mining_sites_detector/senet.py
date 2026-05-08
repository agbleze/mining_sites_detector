import torch
import torch.nn as nn


class SenetStem(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.conv1 = nn.LazyConv2d(out_channels=64, 
                                   kernel_size=7,
                                   stride=2,
                                   padding=3
                                   )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1
                                    )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        return x

class SqueezeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x
        



class ExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.LazyLinear(out_features=None)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.LazyLinear(out_features=None)
        self.sigmoid = nn.Sigmoid()
        
    def initialize_parameters(self, input):
        C = input.shape[1]
        reduced_channels = C // self.reduction_ratio
        
        self.fc1.out_features = reduced_channels
        self.fc2.out_features = C
        
        super().initialize_parameters(input)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x 
    
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.squeeze = SqueezeBlock()
        self.excitation = ExcitationBlock(reduction_ratio=reduction_ratio)
        
        
    def forward(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.excitation(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = shortcut * x
        return x
    
    
class SEResnetProjectionBlock(nn.Module):
    def __init__(self):
        super().__init__()