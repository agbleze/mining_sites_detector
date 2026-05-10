import torch
import torch.nn as nn
from typing import NamedTuple, List
from torchinfo import summary 


def kernel_initializer(m, kernel_initializer="he_normal"):
    if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
                
                

class LazyDepthwiseConv2d(nn.LazyConv2d):
    
    def initialize_parameters(self, input):
        #super().initialize_parameters(input)
        
        print(f"in_channels: {self.in_channels}")
        print(f"out_channels: {self.out_channels}")
        self.groups = self.in_channels
        
        if not self.out_channels:
            self.out_channels = self.in_channels
            
        
        self.weight = nn.Parameter(torch.empty(self.in_channels,
                                               1,
                                               *self.kernel_size,
                                               device=input.device,
                                               dtype=input.dtype
                                               )
                                   )
        
        if self.out_channels != self.in_channels:
            raise ValueError(
                f"Depthwise conv requires out_channels == in_channels, "
                f"got out={self.out_channels}, in={self.in_channels}"
            )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, out_channels, stride, width_multiplier=1):
        super().__init__()
        
        out_channels = int(out_channels * width_multiplier)
        
        self.depthwise_conv = nn.Sequential(LazyDepthwiseConv2d(out_channels=None,
                                                                kernel_size=3,
                                                                stride=stride,
                                                                ),
                                            nn.LazyBatchNorm2d(),
                                            nn.ReLU6(),
                                            )
        self.pointwise_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
                                                          kernel_size=1,
                                                          stride=1
                                                          ),
                                            nn.LazyBatchNorm2d(),
                                            nn.ReLU6()
                                            )
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class MobileNetStem(nn.Module):
    def __init__(self, width_multiplier=1):
        super().__init__()
        self.zeropad = nn.ZeroPad2d(1)
        self.conv1 = nn.LazyConv2d(out_channels=32, kernel_size=3,
                                   stride=2
                                   )
        self.depthwise_separable_conv = DepthwiseSeparableConv(out_channels=64, stride=1,
                                                               width_multiplier=width_multiplier
                                                               )
        
    def forward(self, x):
        x = self.zeropad(x)
        x = self.conv1(x)
        x = self.depthwise_separable_conv(x)
        return x
        

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avg_globalpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1x1 = nn.LazyConv2d(kernel_size=1,
                                     out_channels=num_classes
                                     )
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.avg_globalpool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)
        return x

def group(*, num_blocks, out_channels, width_multiplier, stride=None,
          **kwargs):
    group_blocks = []
    
    for i in range(num_blocks):
        if not stride:
            stride = 2 if i == 0 else 1 # first block is strided
        depthwise_block = DepthwiseSeparableConv(out_channels=out_channels,
                                                 width_multiplier=width_multiplier,
                                                 stride=stride
                                                 )    
        group_blocks.append(depthwise_block)    


def learner(configs, **kwargs):
    learner_groups = []
    
    last_gro_index = len(configs.block_config) - 1
    
    
    for idx, grp_config in enumerate(configs.block_config):
        stride =  2 if idx == last_gro_index else None
        group_block = group(**configs._asdict(), **grp_config._asdict(), stride=stride)
        learner_groups.append(group_block)
        
    return nn.Sequential(*learner_groups)
    
    
def make_model(num_classes, learner_configs, device="cuda"):
    stem = MobileNetStem(width_multiplier=learner_configs.width_multiplier) 
    learner_module = learner(configs=learner_configs)  
    classifier = Classifier(num_classes=num_classes)
    
    model = nn.Sequential(stem, learner_module, classifier)
    model.to(device) 
    return model

        
class MobileNetBlockConfig(NamedTuple):
    out_channels: int
    num_blocks: int
    

class MobileNetGroupConfig(NamedTuple):
    width_multiplier: float
    block_config: List[MobileNetBlockConfig]  
    
    
    
block_config = [MobileNetBlockConfig(out_channels=128, num_blocks=2),
                MobileNetBlockConfig(out_channels=256, num_blocks=2),
                MobileNetBlockConfig(out_channels=512, num_blocks=6),
                MobileNetBlockConfig(out_channels=1024, num_blocks=2)
                ]     



group_configs = MobileNetGroupConfig(width_multiplier=1,
                                     block_config=block_config
                                     )


if __name__ == "__main__":
    device = "cuda"
    data = torch.randn(1, 3, 224, 224).to(device)
    model = make_model(num_classes=1000, learner_configs=group_configs, device=device)
    _ = model(data)
    model.apply(kernel_initializer)
    
    summary(model=model, input_data=data) 
    
"""
the first block depwise con in each block of a group is strided

>> groups share same out_channels /n_filters
>> blocks share similar conv arch


group params mon
width_multiplier,
n-filters
nblocks



"""
        