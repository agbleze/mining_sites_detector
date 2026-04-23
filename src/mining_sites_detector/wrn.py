import torch
import torch.nn as nn
from torchsummary import summary
from .models.utils import kernel_initializer



class WRNStem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem_conv = nn.Sequential(nn.LazyConv2d(out_channels=16, kernel_size=3, 
                                                     stride=1, padding="same"),
                                       nn.LazyBatchNorm2d(),
                                       nn.ReLU()
                                       )
        
        
    def forward(self, x):
        x = self.stem_conv(x)
        return x
    
    
    
class WRNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
        
            

