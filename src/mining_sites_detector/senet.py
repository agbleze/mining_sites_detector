import torch
import torch.nn as nn
from torchinfo import summary
from typing import Literal, NamedTuple, List

VARIANT_OPTIONS = ["standard", "pre", "post", "identity", "enclose"]


def kernel_initializer(m, kernel_initializer="he_normal"):
    if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
                
                
                
class SEResNextStem(nn.Module):
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


class SELazyLinear(nn.LazyLinear):
    def __init__(self, target_out: int = None, 
                 reduction_ratio: int = None, 
                 mode: str = "reduce", 
                 bias: bool = True
                 ):
        super().__init__(out_features=None, bias=bias)
        self.reduction_ratio = reduction_ratio
        #print(f"reduction_ratio: {self.reduction_ratio}")
        self.mode = mode
        self.target_out = target_out  # used for fc2 to know final C
        #print(f"out_features: {self.out_features}")

    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (list, tuple)) else input
        in_ch = int(x.shape[1])
        if self.mode == "reduce":
            out_ch = max(1, in_ch // self.reduction_ratio)
            self.in_features = in_ch
            self.out_features = out_ch
            print(f"IN MODE: {self.mode}")
            print(f"reduction_ratio : {self.reduction_ratio}")
            print(f"in_features: {self.in_features}")
            print(f"out_features: {self.out_features}")
        elif self.mode == "expand":  # expand
            # if target_out provided, use it; otherwise expand to in_ch (fallback)
            print(f"IN MODE: {self.mode}")
            self.in_features = in_ch
            self.out_features = self.target_out if self.target_out is not None else in_ch
            self.out_features = int(self.in_features * self.reduction_ratio)
            print(f"else in_features: {self.in_features}  out_features: {self.out_features}")
        super().initialize_parameters(input)

        
class SqueezeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        print(f"Squeeze block input shape before global_avgpool: {x.shape}")
        x = self.global_avgpool(x)
        print(f"Squeeze block output shape after global_avgpool: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        print(f"Squeeze block output shape after flatten: {x.shape}")
        return x

class ExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = SELazyLinear(reduction_ratio=reduction_ratio, mode="reduce")
        self.relu = nn.ReLU()
        
        self.fc2 = SELazyLinear(target_out=None, mode="expand", reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()
        
        if getattr(self.fc1, "in_features", None) is not None and self.fc2.target_out is None:
            self.fc2.target_out = self.fc1.in_features
        
    def initialize_parameters(self, input):
        print(f"Initializing Excitation block parameters with input shape: {input.shape}")
        x = input[0]
        C = int(x.shape[1])
        print(f"Excitation block input channels: {C}")
        reduced_channels = C // self.reduction_ratio
        print(f"Excitation block reduced channels: {reduced_channels}")
        
        self.fc1.out_features = reduced_channels
        self.fc2.out_features = C
        
        super().initialize_parameters(input)
        
    def forward(self, x):
        print(f"Excitation block input shape before fc1: {x.shape}")
        x = self.fc1(x)
        print(f"Excitation block after fc1: {x.shape}")
        x = self.relu(x)
        x = self.fc2(x)
        print(f"Excitation block after fc2: {x.shape}")
        x = self.sigmoid(x)
        print(f"Excitation block after sigmoid: {x.shape}")
        return x 
    
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.squeeze = SqueezeBlock()
        self.excitation = ExcitationBlock(reduction_ratio=reduction_ratio)
        
        
    def forward(self, x):
        shortcut = x
        print(f"Input shape to SE block: {x.shape}")
        x = self.squeeze(x)
        print(f"Squeeze block output shape: {x.shape}")
        x = self.excitation(x)
        print(f"x after excitation: {x.shape}")
        x = x.unsqueeze(-1).unsqueeze(-1)
        #x = x.view(x.size(0), x.size(1), 1, 1)  # reshape to (B, C, 1, 1)
        print(f"x = x.unsqueeze(-1).unsqueeze(-1) Excitation output after unsqueeze shape: {x.shape}")
        x = shortcut * x
        return x
    
    
class SEResNextProjectionBlock(nn.Module):
    def __init__(self, bottleneck_width, out_channels, cardinality=32, reduction_ratio=16,
                 stride=1,
                 variant: Literal["standard", "pre", "post", "identity"] = "standard"
                 ):
        super().__init__()
        if variant not in VARIANT_OPTIONS:
            raise ValueError(f"Invalid variant option: {variant}. Must be one of {VARIANT_OPTIONS}")
        
        self.variant = variant
        
        bottleneck_channels = bottleneck_width * cardinality
        
        self.projection_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, 
                                                kernel_size=1,
                                                stride=stride, 
                                                padding="same", 
                                                bias=False
                                                ),
                                nn.LazyBatchNorm2d(),
                                )
        
        self.reduction_conv = nn.Sequential(nn.LazyConv2d(out_channels=bottleneck_channels, 
                                                  kernel_size=1, stride=1, 
                                                  padding="same", 
                                                  bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        
        self.group_conv = nn.Sequential(nn.LazyConv2d(out_channels=bottleneck_channels, 
                                                 kernel_size=3, stride=stride, 
                                                 padding="same", bias=False, 
                                                 groups=cardinality
                                                ),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU()
                                    )
        
        self.expansion_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, 
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
    def __init__(self, bottleneck_width, out_channels, cardinality=32, reduction_ratio=16,
                 variant: Literal["standard", "pre", "post", "identity", "enclose"] = "standard"
                 ):
        super().__init__()
        
        if variant not in VARIANT_OPTIONS:
            raise ValueError(f"Invalid variant option: {variant}. Must be one of {VARIANT_OPTIONS}")
        
        self.variant = variant
        
        bottleneck_channels = bottleneck_width * cardinality
        
        self.reduction_conv = nn.Sequential(nn.LazyConv2d(out_channels=bottleneck_channels,
                                                  kernel_size=1, stride=1,
                                                  padding="same", bias=False,
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        self.group_conv = nn.Sequential(nn.LazyConv2d(out_channels=bottleneck_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding="same",
                                                      bias=False,
                                                      groups=cardinality
                                                      ),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU()
                                        )
        
        self.expansion_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
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
    

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
          
        
class ResNextBlocksConfig(NamedTuple):
    out_channels: int
    num_blocks: int    
    
    
class SEResNextGroupsConfig(NamedTuple):
    cardinality: int
    bottleneck_width: int
    reduction_ratio: int
    variant: Literal["standard", "pre", "post", "identity", "enclose"]    
    block_config: List[ResNextBlocksConfig]


def group(*, 
          bottleneck_width, 
          out_channels, 
          num_blocks,
          cardinality=32, reduction_ratio=16,
          variant: Literal["standard", "pre", "post", "identity", "enclose"] = "standard",
          **kwargs
          ):
    block_collection = []
    block = SEResNextProjectionBlock(bottleneck_width=bottleneck_width, out_channels=out_channels,
                                    cardinality=cardinality, reduction_ratio=reduction_ratio,
                                    variant=variant
                                    )
    block_collection.append(block)
    for _ in range(num_blocks -1):
        block = SEResNextIdentityBlock(bottleneck_width=bottleneck_width, out_channels=out_channels,
                                        cardinality=cardinality, reduction_ratio=reduction_ratio,
                                        variant=variant
                                        )
        block_collection.append(block)
    
    return nn.Sequential(*block_collection)


def learner(configs):
    group_blocks = []
    for block_config in configs.block_config:
        grp_blk = group(**configs._asdict(), **block_config._asdict())
        group_blocks.append(grp_blk)
    return nn.Sequential(*group_blocks)


def make_model(num_classes, learner_configs):
    stem = SEResNextStem()
    learner_module = learner(configs=learner_configs)
    classifier = Classifier(num_classes=num_classes)
    
    model = nn.Sequential(stem, learner_module, classifier)
    return model

if __name__ == "__main__":
    
    blocks = [ResNextBlocksConfig(out_channels=256, num_blocks=3),
                ResNextBlocksConfig(out_channels=512, num_blocks=4),
                ResNextBlocksConfig(out_channels=1024, num_blocks=6),
                ResNextBlocksConfig(out_channels=2048, num_blocks=3)
                ]

    group_config = SEResNextGroupsConfig(cardinality=32, bottleneck_width=4, 
                                         reduction_ratio=16,
                                        variant="standard", 
                                        block_config=blocks
                                        )
    
    data = torch.randn(1, 3, 224, 224).to("cuda")
    model = make_model(num_classes=1000, learner_configs=group_config).to("cuda")
    _ = model(data)
    model.apply(kernel_initializer)
    
    
    print(f"Custom SEResNeXt model summary:\n")
    summary(model, input_data=data, device='cuda',
            verbose=1,
            mode="train",
            col_names=["input_size", "output_size", "num_params",
                       "mult_adds",
                       ],
            depth=3,
            )

    
    