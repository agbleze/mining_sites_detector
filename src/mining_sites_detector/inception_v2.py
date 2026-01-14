

import torch
import torch.nn as nn
import numpy as np




class inceptionV2Block(nn.Module):
    def __init__(self, f1xf1, f3xf3, fpool):
        super().__init__()
        
        self.zeropad = nn.ZeroPad2d()
        





