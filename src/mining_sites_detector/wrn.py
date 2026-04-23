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
        
        
            
class WRNIdentityBlock(nn.Module):
    def __init__(self, n_filters, k, dropout_rate, l=1):
        """n_filters: number of filters in the convolutional layers
           k: widening factor
           l: number of convolutional layers in the block
           dropout_rate: dropout rate for regularization
        """
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
        
        conv = nn.Sequential(nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(out_channels=n_filters*k, kernel_size=3),
                                
                                #nn.Dropout(dropout_rate),
                                
                                
                            )
        
        block_conv = []
        
        for _ in range(l):
            block_conv.append(conv)
            block_conv.append(nn.Dropout(dropout_rate))
        
        block_conv = block_conv[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_conv)
        
        
    def forward(self, x):
        shortcut = x
        x = self.block_conv(x)
        x += shortcut
        return x
        
        
        
class WRNProjectionBlock(nn.Module):
    def __init__(self, n_filters, k, dropout_rate, l=2):
        """
        n_filters: number of filters in the convolutional layers
        k: widening factor
        l: number of convolutional layers in the block
        dropout_rate: dropout rate for regularization
        """
        
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
        
        self.proj = nn.Sequential(
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU(),
                                    nn.LazyConv2d(out_channels=n_filters*k, kernel_size=1)
                                )        
        
        block_convs = []
        
        conv = nn.Sequential(nn.LazyBatchNorm2d(),
                             nn.ReLU(),
                             nn.LazyConv2d(out_channels=n_filters*k, kernel_size=3)
                             )
        for _ in range(l):
            block_convs.append(conv)
            block_convs.append(nn.Dropout(dropout_rate))
            
        block_convs = block_convs[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_convs)
        
        
    def forward(self, x):
        shortcut = self.proj(x)
        
        x = self.block_conv(x)
        x += shortcut
        return x
        
        
        
        
    
