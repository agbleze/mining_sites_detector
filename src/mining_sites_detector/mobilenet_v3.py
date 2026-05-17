import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List, Literal


NON_LINEARITY_VALUES = ["relu", "hswish"]
KERNEL_SIZE_VALUES = ["3x3", "5x5"]


class LazyDepthwiseConv2d(nn.LazyConv2d):
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (tuple, list)) else input
        device = x.device
        dtype = x.dtype
        
        in_ch = x.shape[1]
        if not self.out_channels:
            self.out_channels = int(in_ch)
            
        self.in_channels = int(in_ch)
        self.groups = int(in_ch)
        
        if isinstance(self.kernel_size, int):
            k = (self.kernel_size, self.kernel_size)
        else:
            k = self.kernel_size
            
        weight_shape = (self.out_channels, self.in_channels // self.groups, *k)
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=dtype, device=device))
        
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        

class MobileNetV3Stem(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        self.conv = nn.LazyConv2d(kernel_size=3, out_channels=out_channels,
                                  bias=False,
                                  stride=2
                                  )
        self.hswish = nn.Hardswish()
        self.bn = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        return x
    
    
        
class MobileNetV3InventedResidualBlock(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_rate,
                 non_linearity: Literal["relu", "hswish"],
                 stride, depthwiseconv_kernel_size: Literal["3x3", "5x5"],
                 use_squeeze_excitation: bool,
                 **kwargs
                ):
        super().__init__()
        out_channels = int(out_channels * width_multiplier) if width_multiplier else out_channels
        expanded_channels = int(out_channels * expansion_rate)
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expanded_channels, kernel_size=1,
                                            padding=1, bias=False
                                            )
        
        if depthwiseconv_kernel_size == "3x3":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, 
                                                      kernel_size=3,
                                                      bias=False,
                                                      stride=1,
                                                      padding=1
                                                    )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
            
        elif depthwiseconv_kernel_size == "5x5":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, kernel_size=5,
                                                      stride=1,
                                                      padding=1
                                                      )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=1,
                                            padding=1, bias=False
                                            )
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
            self.act2 = nn.ReLU6()
            self.act3 = nn.ReLU6()
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            self.act2 = nn.Hardswish()
            self.act3 = nn.Hardswish()
            
    def forward(self, x):
        shortcut = x
        x = self.expansion_conv(x)
        x = self.bn1(x)
        self.act1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        #x = self.act3(x)
        
        x += shortcut
        return x
    
    
class MobileNetV3InventedNonResidualBlock(nn.Module):
    def __init__(self, out_channels, width_multipler, expansion_rate,
                 non_linearity: Literal["relu", "hswish"],
                depthwiseconv_kernel_size: Literal["3x3", "5x5"],
                use_squeeze_excitation: bool,
                stride=2,
                **kwargs
                ):
        super().__init__()
        if non_linearity not in NON_LINEARITY_VALUES:
            raise ValueError(f"non_linearity: {non_linearity} was provided but expected to be one of {NON_LINEARITY_VALUES}")
        
        if depthwiseconv_kernel_size not in KERNEL_SIZE_VALUES:
            raise ValueError(f"depthwiseconv_kernel_size: {depthwiseconv_kernel_size} is not a valid value. It must be one of {KERNEL_SIZE_VALUES}")
        
        out_channels = int(out_channels * width_multipler) if width_multipler else out_channels
        expanded_channels = int(out_channels * expansion_rate)
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expanded_channels,
                                            kernel_size=1, padding=1,
                                            bias=False
                                            )
        if depthwiseconv_kernel_size == "3x3":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, 
                                                      kernel_size=3,
                                                      stride=stride,
                                                      bias=False, padding=1
                                                      )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
        elif depthwiseconv_kernel_size == "5x5":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                      kernel_size=5,
                                                      stride=stride,
                                                      padding=1, bias=False
                                                      )
            
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
                
            
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            padding=1, bias=False
                                            )
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
            self.act2 = nn.ReLU6()
            self.act3 = nn.ReLU6()
            
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            self.act2 = nn.Hardswish()
            self.act3 = nn.Hardswish()
            
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
            
    def forward(self, x):
        x = self.expansion_conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        #x = self.act3(x)
        return x
 
 
class SELazyLinear(nn.LazyLinear):
    def __init__(self, reduction_ratio,
                 mode: Literal["reduce", "expand"],
                 bias=True
                 ):
        super().__init__(out_features=None)
        self.reduction_ratio = reduction_ratio
        self.mode = mode
        
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (list, tuple)) else input
        
        in_ch = int(x.shape[0])
        self.in_features = in_ch
                
        if self.mode == "reduce":
            self.out_features = int(max(1, in_ch // self.reduction_ratio))
        elif self.mode == "expand":
            self.out_features = int(self.in_features * self.reduction_ratio)
        super().initialize_parameters()

        
               
class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
    def forward(self, x):
        x = self.global_avgpool(x)
        return x
        
                    
class Excitation(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.fc1 = SELazyLinear(reduction_ratio=reduction_ratio, 
                                mode="reduce",
                               bias=False
                               )
        self.relu = nn.ReLU()   
        self.fc2 = SELazyLinear(reduction_ratio=reduction_ratio,
                                mode="expand", 
                                bias=False
                                )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
             
        

class SqueezeExcitation(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.squeeze = Squeeze()
        self.excitation = Excitation(reduction_ratio=reduction_ratio)
        self.calibrate = nn.Sigmoid()
        
    def forward(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.excitation(x)
        x = self.calibrate(x)
        recalibrated_x = shortcut * x
        return recalibrated_x


class MobileNetV3LastLearnerBlock(nn.Module):
    def __init__(self, out_channels, non_linearity="hswish"):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv = nn.LazyConv2d(out_channels=out_channels,
                                  kernel_size=1,
                                  bias=False
                                  )
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            
    def forward(self, x):
        x = self.global_avgpool(x)
        x = self.conv(x)
        x = self.act1(x)
        return x
        
        
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=num_classes,
                                  kernel_size=1,
                                  bias=False
                                  )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
    

class BlockConfig(NamedTuple):
    num_blocks: int
    out_channels: int
    non_linearity: Literal[NON_LINEARITY_VALUES]    
    use_squeeze_excitation: bool
    invented_residual: bool = True
    batchnorm = True
    expansion_sizes: List
    
class MobileNetV3GroupConfig(NamedTuple):
    width_multiplier: int
    #expansion_rate: int
    block_config: List(BlockConfig)
    
    
def group(*, out_channels, width_multiplier,
          expansion_rate, non_linearity,
          use_squeeze_excitation,
          num_blocks,
          **kwargs,
          ):
    blocks = []
    
    for i in num_blocks:
        pass
    
    
    
    
block_config = [BlockConfig(out_channels=16, num_blocks=1, non_linearity="relu", use_squeeze_excitation=False, expansion_sizes=[16]),
          