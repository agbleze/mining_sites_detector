import torch
import torch.nn as nn
from torchsummary import summary
#from .models.utils import kernel_initializer



class WRNStem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem_conv = nn.Sequential(nn.LazyConv2d(out_channels=16, kernel_size=3, 
                                                     #stride=1, #padding="same", 
                                                     bias=False
                                                     ),
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
    def __init__(self, n_filters, #k, 
                 dropout_rate, l, stride=None):
        """n_filters: number of filters in the convolutional layers
           k: widening factor
           l: number of convolutional layers in the block
           dropout_rate: dropout rate for regularization
        """
        self.n_filters = n_filters
        #self.k = k
        self.dropout_rate = dropout_rate
        self.l = l
        
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
  
        
        block_conv = []
        
        for _ in range(l):
            if _ == 0:
                stride = 1 #2 if not stride else stride
                conv = basic_conv(n_filters=n_filters, #k=k,
                                  stride=stride)
            else:
                conv = basic_conv(n_filters=n_filters, #k=k,
                                  stride=1)
            block_conv.append(conv)
            block_conv.append(nn.Dropout(dropout_rate))
        
        block_conv = block_conv[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_conv)
           
    
    def forward(self, x):
        shortcut = x
        x = self.block_conv(x)
        x += shortcut
        return x
        

def basic_conv(n_filters,#k, 
               stride):
    #out_channels = n_filters * k
    print(f"Basic convolution: {n_filters} channels")
    conv = nn.Sequential(nn.LazyBatchNorm2d(),
                        nn.ReLU(),
                        nn.LazyConv2d(out_channels=n_filters, #out_channels, 
                                        kernel_size=3, stride=stride, 
                                        padding=1, bias=False
                                        ),
                        #nn.Dropout(dropout_rate),                                
                        )
    return conv
        
        
class WRNProjectionBlock(nn.Module):
    def __init__(self, n_filters, #k, 
                 dropout_rate, l=2, stride=None):
        """
        n_filters: number of filters in the convolutional layers
        k: widening factor
        l: number of convolutional layers in the block
        dropout_rate: dropout rate for regularization
        """
        
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
        
        self.proj = nn.Sequential(nn.LazyBatchNorm2d(),
                                #nn.ReLU(),
                                nn.LazyConv2d(out_channels=n_filters, kernel_size=1,
                                                stride=2 if not stride else stride, #stride if stride else 1, 
                                                padding=0, bias=False
                                                )
                                )        
        
        block_convs = []
        
        # conv = nn.Sequential(nn.LazyBatchNorm2d(),
        #                      nn.ReLU(),
        #                      nn.LazyConv2d(out_channels=n_filters*k, kernel_size=3)
        #                      )
        for _ in range(l):
            if _ == 0:
                stride = 2 if not stride else stride
                conv = basic_conv(n_filters=n_filters, #k=k, 
                                  stride=stride)
            else:
                conv = basic_conv(n_filters=n_filters, #k=k, 
                                  stride=1)
                
            block_convs.append(conv)
            block_convs.append(nn.Dropout(dropout_rate))
            
        block_convs = block_convs[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_convs)
        
        
    def forward(self, x):
        shortcut = self.proj(x)
        
        x = self.block_conv(x)
        x += shortcut
        return x
        

def group(out_features, #k, 
          n_blocks, dropout_rate, l, stride=None):
    """
    out_features: number of filters in the convolutional layers
    k: widening factor
    n_blocks: number of blocks in the group
    dropout_rate: dropout rate for regularization
    l: number of convolutional layers in each block
    """
    #out_channels = out_features * k
    print(f"Group with {n_blocks} blocks, each with {l} convolutional layers, and {out_features} output channels.")
    block_collection = []
    proj_conv = WRNProjectionBlock(n_filters=out_features, #k=k, 
                                   dropout_rate=dropout_rate, l=l, stride=stride)
    
    block_collection.append(proj_conv)
    for _ in range(n_blocks -1):
        block = WRNIdentityBlock(n_filters=out_features, #k=k, 
                                 dropout_rate=dropout_rate, l=l)
        block_collection.append(block)
    print(f"Number of blocks in the group: {len(block_collection)}")
    return nn.Sequential(*block_collection)


def learner(groups, depth=40):
    n_blocks = (depth -2) // 6 
    print(f"Number of blocks per group: {n_blocks}")
    learner_grps_collection = [] 
    for _, params in enumerate(groups):
        if _ == 0:
            stride = 1
            n_filters, k, dropout_rate, l = params
            n_filters = n_filters * k
            grp_blk = group(out_features=n_filters, #k=k, 
                            n_blocks=n_blocks, dropout_rate=dropout_rate, l=l, 
                            stride=stride
                            )
            learner_grps_collection.append(grp_blk)
        else:
            n_filters, k, dropout_rate, l = params
            n_filters = n_filters * k
            grp_blk = group(out_features=n_filters, #k=k,
                            n_blocks=n_blocks, dropout_rate=dropout_rate, l=l)
            learner_grps_collection.append(grp_blk)
    return nn.Sequential(*learner_grps_collection)
            
     
         
     
        
group_config = [(16, 4,  0.3, 2), (32, 4, 0.3, 2), (64, 4, 0.3, 2)]   


import torchvision.models as models
from torchsummary import summary


# model = models.wide_resnet50_2(weights=None).to("cuda")
# summary(model, input_size=(3, 224, 224))


if __name__ == "__main__":
    
    def kernel_initializer(m, kernel_initializer="he_normal"):
        if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if kernel_initializer == "he_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif kernel_initializer == "glorot_uniform":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    example_input = torch.randn(1, 3, 224, 224, device="cuda")
    stem = WRNStem()
    learner_module = learner(groups=group_config, depth=40)
    classifier = WRNClassifier(num_classes=1000)
    
    model = nn.Sequential(stem, learner_module, classifier).to("cuda")
    _ = model(example_input)
    model.apply(kernel_initializer)
    
    print(f"Custom WRN model summary:\n{summary(model, input_size=(3, 32, 32))}")