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
    
    
class SequeezeExcitationBlock(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 1x1 x C/r
        # need to infer the number of channels in the input feature map 
        # to determine the output features for this layer
        self.fc1 = nn.LazyLinear(out_features=r)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.LazyLinear()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        shortcut = x
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        self.relu(x)
        self.fc2(x)
        x = self.sigmoid(x)
        x = shortcut * x 
        return x     