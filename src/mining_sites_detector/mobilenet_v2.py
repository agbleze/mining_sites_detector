import torch
import torch.nn as nn



class MobileNetStemV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=32, kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  stride=2
                                  )
        
        
    def forward(self, x):
        x = self.conv(x)
        return x
        

class MobileNet_V2(nn.Module):
    def __init__(self, width_multiplier):
        pass
    
    
    