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
    
    
    
    

