import torch
import torch.nn as nn
from typing import Literal



VARIANT_OPTIONS = ["standard", "pre", "post", "identity", "enclose"]

class SenetStem(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.conv1 = nn.LazyConv2d(out_channels=64, 
                                   kernel_size=7,
                                   stride=2,
                                   padding=3
                                   )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1
                                    )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        return x

class SqueezeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x
        



class ExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.LazyLinear(out_features=None)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.LazyLinear(out_features=None)
        self.sigmoid = nn.Sigmoid()
        
    def initialize_parameters(self, input):
        C = input.shape[1]
        reduced_channels = C // self.reduction_ratio
        
        self.fc1.out_features = reduced_channels
        self.fc2.out_features = C
        
        super().initialize_parameters(input)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x 
    
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.squeeze = SqueezeBlock()
        self.excitation = ExcitationBlock(reduction_ratio=reduction_ratio)
        
        
    def forward(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.excitation(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = shortcut * x
        return x
    
    
class SEResNextProjectionBlock(nn.Module):
    def __init__(self, filters_in, filters_out, cardinality=32, reduction_ratio=16,
                 stride=1,
                 variant: Literal["standard", "pre", "post", "identity"] = "standard"
                 ):
        super().__init__()
        if variant not in VARIANT_OPTIONS:
            raise ValueError(f"Invalid variant option: {variant}. Must be one of {VARIANT_OPTIONS}")
        
        self.variant = variant
        
        self.projection_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_out, 
                                                kernel_size=1,
                                                stride=stride, 
                                                padding="same", 
                                                bias=False
                                                ),
                                nn.LazyBatchNorm2d(),
                                )
        
        self.reduction_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_in, 
                                                  kernel_size=1, stride=1, 
                                                  padding="same", 
                                                  bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        
        self.group_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_in, 
                                                 kernel_size=3, stride=stride, 
                                                 padding="same", bias=False, 
                                                 groups=cardinality
                                                ),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU()
                                    )
        
        self.expansion_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_out, 
                                                  kernel_size=1, stride=1, 
                                                  padding="same", bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    )
        
        self.act = nn.ReLU()
        self.residual_block = nn.Sequential(self.reduction_conv,
                                            self.group_conv,
                                            self.expansion_conv
                                            )
        self.se_block = SqueezeExcitationBlock(reduction_ratio=reduction_ratio)
        
    def forward(self, x):
        shortcut = self.projection_conv(x)
        
        if self.variant == "standard":
            x = self.residual_block(x)
            x = self.se_block(x)
            x += shortcut
            
        elif self.variant == "pre":
            x = self.se_block(x)
            x = self.residual_block(x)
            x += shortcut
            
        elif self.variant == "post":
            x = self.residual_block(x)
            x += shortcut
            x = self.se_block(x)
            
        elif self.variant == "identity":
            x_se = self.se_block(x)
            x_residual = self.residual_block(x)
            x = x_se + x_residual
            
        x = self.act(x)    
        return x
    
    
        
class SEResNextIdentityBlock(nn.Module):
    def __init__(self, filters_in, filters_out, cardinality=32, reduction_ratio=16,
                 variant: Literal["standard", "pre", "post", "identity", "enclose"] = "standard"
                 ):
        super().__init__()
        
        if variant not in VARIANT_OPTIONS:
            raise ValueError(f"Invalid variant option: {variant}. Must be one of {VARIANT_OPTIONS}")
        
        self.variant = variant
        
        self.reduction_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_in,
                                                  kernel_size=1, stride=1,
                                                  padding="same", bias=False,
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        self.group_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_in,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding="same",
                                                      bias=False,
                                                      groups=cardinality
                                                      ),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU()
                                        )
        
        self.expansion_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_out,
                                                       kernel_size=1, stride=1,
                                                       padding="same", bias=False
                                                       ),
                                         nn.LazyBatchNorm2d(),
                                         )
        
        self.act = nn.ReLU()
        
        self.residual_block = nn.Sequential(self.reduction_conv,
                                            self.group_conv,
                                            self.expansion_conv
                                            )
        self.se_block = SqueezeExcitationBlock(reduction_ratio=reduction_ratio)
        
        
    def forward(self, x):
        shortcut = x
        
        if self.variant == "standard":
            x = self.residual_block(x)
            x = self.se_block(x)
            x += shortcut
            
        elif self.variant == "pre":
            x = self.se_block(x)
            x = self.residual_block(x)
            x += shortcut
            
        elif self.variant == "post":
            x = self.residual_block(x)
            x += shortcut
            x = self.se_block(x)
            
        elif self.variant == "identity":
            x_se = self.se_block(x)
            x_residual = self.residual_block(x)
            x = x_se + x_residual
            
        elif self.variant == "enclose":
            x = self.reduction_conv(x)
            x = self.group_conv(x)
            x = self.se_block(x)
            x = self.expansion_conv(x)
            x += shortcut
            
        x = self.act(x)    
        return x