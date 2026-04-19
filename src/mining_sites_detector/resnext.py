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
        x = torch.flatten(x, dim=1)
        x = self.softmax(x)
        return x
        
        
class ResNextIdentityBlock(nn.Module):
    def __init__(self, filter_in, filter_out, cardinality=32):
        super(self).__init__(self) 
        
        # 1x1 Dimensionality reduction
        self.reduce = nn.Sequential(
                                    nn.LazyConv2d(out_channels=filter_in, kernel_size=1, stride=1, padding="same", bias=False),
                                    nn.BatchNorm2d(),
                                    nn.ReLU()
                                )
        
        # Cardinality (wide) Layer split-transform
        self.group = nn.Sequential(nn.LazyConv2d(out_channels=filter_in, kernel_size=3, stride=1, padding="same", bias=False, 
                                                 groups=cardinality
                                                ),
                                   nn.LazyBatchNorm2d(),
                                   )
        self.relu = nn.ReLU()
        
        # 1x1 expansion 
        self.expand = nn.Sequential(nn.LazyConv2d(out_channels=filter_out,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding="same", bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        
        
    def forward(self, x):
        shortcut = x
        x = self.reduce(x)
        x = self.group(x)
        x = self.expand(x)
        x += shortcut
        x = self.relu(x)
        return x
        

class ResNextProjectionBlock(nn.Module):
    def __init__(self, filters_in, filters_out, cardinality=32, strides=2, **kwargs):
        super().__init__(**kwargs)
        # construct projection shortcut layer
        self.proj = nn.Sequential(nn.LazyConv2d(out_channels=filters_out, kernel_size=1,
                                                stride=strides, padding="same",
                                                ),
                                  nn.LazyBatchNorm2d(),
                                  )
        
    