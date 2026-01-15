

import torch
import torch.nn as nn
import numpy as np




class inceptionV2Block(nn.Module):
    def __init__(self, f1xf1, f3xf3, fpool):
        super().__init__()
        
        self.zeropad = nn.ZeroPad2d()
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.conv1x1_a = nn.Conv2d(kernel_size=1, stride=1, padding="same")
        





