

#%%
import torch
import torch.nn as nn

class StemDenseNet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=2*out_channels,
                                   kernel_size=7, stride=2,
                                   bias=False
                                   ) 
        self.bn1 = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  
        
    def get_zeropadding(self, padding):
        return nn.ZeroPad2d(padding=padding)
    
    def forward(self, x):
        x = self.get_zeropadding(paidding=3)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.get_zeropadding(padding=1)(x)
        x = self.pool(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        
        # BN-RE-Conv 1x1
        # demensionality expansion, expand filters by 4 (DenseNet-B)
        self.bn1 = nn.LazyBatchNorm2d()
        self.act1 = nn.ReLU()
        self.conv1 = nn.LazyConv2d(out_channels=4 * n_filters,
                                   kernel_size=1,
                                   stride=1,
                                   bias=False,
                                   )
        
        # BN-RE-Conv 3x3 with padding=same to preserve same shape of feature maps
        self.bn2 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(out_channels=n_filters,
                                   kernel_size=3, stride=1,
                                   bias=False, padding=1,
                                   )
        self.act2 = nn.ReLU()
        
    def forward(self, x):
        shortcut = x

        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        
        x = torch.cat(shortcut, x, dim=1)
        return x
        


class TransBlock(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        n_filters = int(int(x.shape[3]) * reduction)
        
        # BN-LI-Conv pre-activation 1x1
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv1 = nn.LazyConv2d(out_channels=n_filters,
                                   kernel_size=1, stride=1,
                                   bias=False
                                   )
        self.avgpool = nn.AvgPool2d(kernel_size=2,
                                    stride=2
                                    )
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x
    
    
def group_densenet(n_blocks, n_filters, reduction=None):
    # construct group of dense blocks
    block_collection = []
    
    for _ in range(n_blocks):
        dense_block = DenseBlock(n_filters=n_filters)
        block_collection.append(dense_block)
        
    if reduction:
        trans_block = TransBlock(reduction=reduction)
        block_collection.append(trans_block)
    
    return nn.Sequential(**block_collection)
        

def learner_densenet(groups, n_filters, reduction):
    last = groups.pop()
    group_collection = []
    for n_blocks in groups:
        grp = group_densenet(n_blocks=n_blocks,
                             n_filters=n_filters,
                             reduction=reduction
                             )
        group_collection.append(grp)
    # last group without transition block
    group_collection.append(group_densenet(n_blocks=last,
                                           n_filters=n_filters,
                                           reduction=None
                                           )
                            )
    return nn.Sequential(**group_collection)



    
class ClassifierDenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.act = nn.Softmax(dim=1)
    
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, dim=1)
        x = self.fc(x)
        x = self.act(x)
        return x
        
         
# %%
