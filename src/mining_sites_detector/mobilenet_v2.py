import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List
from dataclasses import dataclass
from copy import deepcopy


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
        
        x = input[0] if isinstance(input, (tuple, list)) else input
        device = x.device
        dtype = x.dtype
        
        in_ch = int(x.shape[1])
        
        out_ch = getattr(self, "out_channels", None)
        
        if out_ch is None or out_ch == 0:
            out_ch = in_ch
        
        self.groups = in_ch
        self.out_channels = in_ch
        self.in_channels = in_ch
        
        k = self.kernel_size
        if isinstance(k, int):
            k = (k, k)
            
                
        if self.in_channels != self.out_channels:
            raise ValueError(f"Depthwise conv expects in_channels == out_channels but got in_channels {self.in_channels} != {self.in_channels}")
        
        
        weight_shapee = (self.out_channels, self.in_channels // self.groups, *k)
        
        self.weight = nn.Parameter(torch.empty(weight_shapee,device=device, dtype=dtype))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        else:
            self.bias = None
        
  

class MobileNetV2Stem(nn.Module):
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
        

class MobileNetResidualBlock_V2(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_rate):
        super().__init__()
        
        out_channels = int(width_multiplier * out_channels)
        expanded_channels = int(out_channels * expansion_rate)
        self.conv1x1 = nn.LazyConv2d(out_channels=expanded_channels,
                                     kernel_size=1,
                                     bias=False,
                                     stride=1
                                     )
        self.relu6_1 = nn.ReLU6()
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3,
                                                  bias=False,
                                                  padding=1
                                                  )
        self.relu6_2 = nn.ReLU6()
        
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            bias=False
                                            )
        
    def forward(self, x):
        shortcut = x
        x = self.conv1x1(x)
        x = self.relu6_1(x)
        
        x = self.depthwise_conv(x)
        x = self.relu6_2(x)
        x = self.pointwise_conv(x)
        
        x += shortcut
        return x
    
    
class MobileNetNonResidualBlock_V2(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_rate):
        super().__init__()
        out_channels = int(out_channels * width_multiplier)
        
        expanded_channels = int(out_channels * expansion_rate)
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expanded_channels,
                                            kernel_size=1,
                                            bias=False,
                                            )
        self.relu6_1 =nn.ReLU6()
        
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  bias=False
                                                  )
        self.relu6_2 = nn.ReLU6()
        
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            bias=False
                                            )
        
    def forward(self, x):
        x = self.expansion_conv(x)
        x = self.relu6_1(x)
        
        x = self.depthwise_conv(x)
        x = self.relu6_2(x)
        x = self.pointwise_conv(x)
        return x
        


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1x1 = nn.LazyConv2d(out_channels=num_classes,
                                     kernel_size=1
                                     )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)
        return x

def group(*, num_blocks, out_channels, width_multiplier, expansion_rate, **kwargs):
    
    grp_blks = []
    for i in range(num_blocks):
        if i == 0:
            blk = MobileNetResidualBlock_V2(out_channels=out_channels,
                                      width_multiplier=width_multiplier,
                                      expansion_rate=expansion_rate
                                      )
        else:
            blk = MobileNetNonResidualBlock_V2(out_channels=out_channels,
                                               width_multiplier=width_multiplier,
                                               expansion_rate=expansion_rate
                                               )
        grp_blks.append(blk)
        
    return nn.Sequential(*grp_blks)


class BlockConfig(NamedTuple):
    out_channels: int
    num_blocks: int        
    
class MobileNetV2GroupConfig(NamedTuple):
    width_multiplier: float
    expansion_rate: int
    block_config: List[BlockConfig]
    
    
    
def learner(configs: MobileNetV2GroupConfig):
    learner_groups = []
    for idx, conf in enumerate(configs.block_config):
        if idx == 0:
            config_0 = deepcopy(configs)._asdict()
            config_0["expansion_rate"] = 1
        
            grp = group(**config_0, **conf._asdict())
            
            learner_groups.append(grp)
        else:
            grp = group(**configs._asdict(), **conf._asdict())
            learner_groups.append(grp)
            
    return nn.Sequential(*learner_groups)
    

def make_model(num_classes, learner_config, device="cuda"):
    stem = MobileNetV2Stem()
    learner_module = learner(learner_config)
    classifier = Classifier(num_classes=num_classes)
    model = nn.Sequential(stem, learner_module, classifier)
    model.to(device)
    return model
    
    
if __name__ == "__main__":
    block_config = [BlockConfig(out_channels=16, num_blocks=1),
                    BlockConfig(out_channels=24, num_blocks=2),
                    BlockConfig(out_channels=32, num_blocks=3),
                    BlockConfig(out_channels=64, num_blocks=4),
                    BlockConfig(out_channels=96, num_blocks=3),
                    BlockConfig(out_channels=160, num_blocks=3),
                    BlockConfig(out_channels=320, num_blocks=1),
                    BlockConfig(out_channels=1280, num_blocks=1)
                    ]
    
    
    learner_config = MobileNetV2GroupConfig(width_multiplier=1,
                                            expansion_rate=6,
                                            block_config=block_config
                                            )
    
    device = "cuda"
    data = torch.randn((3, 3, 224, 224)).to(device)
    model = make_model(num_classes=1000, learner_config=learner_config,
                       device=device
                       )
    _ = model(data)
    model.apply(kernel_initializer)
    
    summary(model=model, input_data=data,
            col_names=["input_size", "output_size", "num_params",
                       "mult_adds",
                       ],
            depth=4
            )
    
    
    
    
