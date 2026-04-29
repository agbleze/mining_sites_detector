import torch
import torch.nn as nn
from torchsummary import summary



class XceptionStem(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        
        self.conv1 = nn.Sequential(
                                nn.LazyConv2d(n_filters, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(inplace=True)
                            )
        
        