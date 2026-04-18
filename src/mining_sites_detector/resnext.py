import torch.nn as nn
import torch


class ResNextStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding="same")
        self.bn = nn.BatchNorm2d()
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)
        return x
        

class ResNextClassifier(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.global_avgpool = nn.AvgPool2d(kernel_size=1)
        self.fc = nn.LazyLinear(out_features=num_classes)   
        self.softmax = nn.Softmax(dim=1)     
        
    def forward(self, x):
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.softmax(x)
        return x
        
        
class ResNextIdentityBlock(nn.Module):
    def __init__(self):
        super(self).__init__(self) 
        
               