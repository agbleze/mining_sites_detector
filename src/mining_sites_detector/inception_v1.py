

#%%
import torch
import torch.nn as nn

from models.utils import kernel_initializer


#%%
class NaiveInceptionModule(nn.Module):
    def __init__(self,):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding="same")
        self.conv1 = nn.LazyConv2d(out_channels=64, stride=1, kernel_size=1, padding="same")
        self.conv2 = nn.LazyConv2d(out_channels=96, stride=1, kernel_size=3, padding="same")
        self.conv3 = nn.LazyConv2d(out_channels=48, stride=1, kernel_size=5, padding="same")
        self.act = nn.ReLU()
        
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.act(self.conv1(x))
        x3 = self.act(self.conv2(x))
        x4 = self.act(self.conv3(x))
        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output
        
        
class Inception1Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding="same")
        self.conv1x1_a = nn.LazyConv2d(out_channels=64, stride=1, kernel_size=1, padding="same")
        self.conv1x1_b = nn.LazyConv2d(out_channels=64, stride=1, kernel_size=1, padding="same")
        self.conv1x1_c = nn.LazyConv2d(out_channels=64, stride=1, kernel_size=1, padding="same")
        self.conv3x3 = nn.LazyConv2d(out_channels=96, kernel_size=3, stride=1, padding="same")
        self.conv1x1_d = nn.LazyConv2d(out_channels=64, stride=1, kernel_size=1, padding="same")
        self.conv5x5 = nn.LazyConv2d(out_channels=48, kernel_size=5, stride=1, padding="same")
        self.act = nn.ReLU()
        
    def forward(self, x):
        x1 = self.maxpool(x)
        x1 = self.act(self.conv1x1_a(x1))

        x2 = self.act(self.conv1x1_b(x))

        x3 = self.act(self.conv1x1_c(x))
        x3 = self.act(self.conv3x3(x3))

        x4 = self.act(self.conv1x1_d(x))
        x4 = self.act(self.conv5x5(x4))
        
        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output
        
        
        
class InceptionStem(nn.Module):
    def __init__(self, #out_channels
                 ):
        super().__init__()
        self.conv1_7x7 = nn.LazyConv2d(out_channels=64, stride=2, kernel_size=7, padding="valid") 
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_1x1 = nn.LazyConv2d(out_channels=64, kernel_size=1, stride=1, padding="same")
        self.conv3_3x3 = nn.LazyConv2d(out_channels=192, kernel_size=3, stride=1, padding="valid")
     
    def zeropad(self, padding):
        return nn.ZeroPad2d(padding=padding)
        
    def forward(self, x):
        x = self.zeropad(padding=3)(x)
        x = self.act(self.conv1_7x7(x))
        x = self.zeropad(padding=1)(x)
        x = self.maxpool(x)
        
        x = self.act(self.conv2_1x1(x))
        x = self.zeropad(padding=1)(x)
        x = self.act(self.conv3_3x3(x))
        
        x = self.zeropad(padding=1)(x)
        x = self.maxpool(x)
        return x
              
    
class InceptionBlock(nn.Module):
    def __init__(self, f1x1, f3x3, f5x5, fpool):
        super().__init__()
        
        self.act = nn.ReLU()
        self.zeropad = nn.ZeroPad2d(padding=1)
        
        self.b1x1 = nn.LazyConv2d(out_channels=f1x1[0], kernel_size=1, padding="same")
           
        self.b3x3_1 = nn.LazyConv2d(out_channels=f3x3[0], kernel_size=1, stride=1, padding="same")
        self.b3x3_2 = nn.LazyConv2d(out_channels=f3x3[1], kernel_size=3, stride=1, padding="valid")
        
        self.b5x5_1 = nn.LazyConv2d(out_channels=f5x5[0], kernel_size=1, padding="same")
        self.b5x5_2 = nn.LazyConv2d(out_channels=f5x5[1], kernel_size=3, stride=1, padding="valid")
        
        self.bpool = nn.MaxPool2d(kernel_size=3, stride=1, padding="same")
        self.bpool_conv1x1 = nn.LazyConv2d(out_channels=fpool[0], kernel_size=1, padding="same")
        
    def forward(self, x):
        # 1x1 branch
        x_b1xb1 = self.act(self.b1x1(x))
        
        # 3 x 3 branch
        # 1 x 1 reduction
        x_b3xb3 = self.act(self.b3x3_1(x))
        x_b3xb3 = self.zeropad(x_b3xb3)
        x_b3xb3 = self.act(self.b3x3_2(x_b3xb3))
        
        # 5 x 5 branch
        # 1 x 1 reduction
        x_b5x5 = self.act(self.b5x5_1(x))
        x_b5x5 = self.zeropad(x_b5x5)
        x_b5x5 = self.act(self.b5x5_2(x_b5x5))
        
        # pooling branch
        x_bpool = self.bpool(x)
        x_bpool = self.act(self.bpool_conv1x1(x_bpool))
        
        output = torch.cat([x_b1xb1, x_b3xb3, x_b5x5, x_bpool], dim=1)
        return output
            
        
class InceptionAuxiliaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv_1x1 = nn.LazyConv2d(out_channels=128, kernel_size=1, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(out_features=1024)
        self.fc2 = nn.LazyLinear(out_features=num_classes) 
        self.act = nn.ReLU()   
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.act(self.conv_1x1(x))
        x = torch.flatten(x, dims=1)
        x = self.act(self.fc1(x))
        x = nn.Dropout(0.7)(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
       

class InceptionClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.dropout_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, dims=1)
        x = nn.Dropout(p=self.dropout_rate)(x)
        
        # final dense layer
        x = self.fc(x)
        x = self.softmax(x)
        return x




def group_inception_v1(#x, 
                       blocks, pooling=True, n_classes=1000):
    """
    Docstring for group_inception
    
    :param x: inputs
    :param blocks: filters configuration for each inception block in the group
    :param pooling: whether to apply maxpooling at end of the group
    :param n_classes: number of classes for auxiliary classifier
    """
    
    aux = []
    incepblk = []
    
    for idx, block in enumerate(blocks):
        if block is None:
            aux.append(InceptionAuxiliaryClassifier(#x, 
                                                    n_classes=n_classes))
        else:
            #x = InceptionBlock(x, block[0], block[1], block[2], block[3])
            incepblk.append(InceptionBlock(block[0], block[1], block[2], block[3]))
            
    if pooling:
        #x = nn.ZeroPad2d(padding=1)(x)
        #x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        incepblk.append(nn.ZeroPad2d(padding=1))
        incepblk.append(nn.MaxPool2d(kernel_size=3, stride=2))
    #return x, aux
    return incepblk, aux



def learner_inception_v1(x, n_classes, group_params):
    """
    Docstring for learner_inception_v1
    
    :param x: inputs
    :param n_classes: number of classes
    :param group_params: list of group parameters
    """
    
    group3_param = [((64,),  (96,128),   (16, 32), (32,)),  # 3a
                     ((128,), (128, 192), (32, 96), (64,)) # 3b
                    ]
    
    group4_param = [((192,),  (96, 208), (16, 48), (64,)), # 4a
                     None, 				 # auxiliary classifier
                     ((160,), (112, 224), (24, 64), (64,)), # 4b
                     ((128,), (128, 256), (24, 64), (64,)), # 4c
                     ((112,), (144, 288), (32, 64), (64,)), # 4d
                     None,                                  # auxiliary classifier
                     ((256,), (160, 320), (32, 128), (128,)) # 4e
                     ] 
    
    group5_param = [((256,), (160, 320), (32, 128), (128,)), # 5a
                     ((384,), (192, 384), (48, 128), (128,))
                    ]# 5b
    
    
    aux = []
    x, o = group_inception_v1(#x,
                              group3_param)
    aux += o
    
    x, o = group_inception_v1(#x, 
                              group4_param, n_classes=n_classes)
    aux += o
    
    x, o = group_inception_v1(#x, 
                              group5_param, pooling=False)
    aux += o
    return x, aux    
    
    
    
# class LearnerInception_v1(nn.Module):
#     def __init__(self, group_params, num_classes):
#         super().__init__()
        
#         learner = []
#         for param in group_params:
#             incepgroup = Inceptionv1Group(blocks=param, num_classes=num_classes)
#             learner.append(incepgroup)
            
#         return nn.Sequential(**learner)


class InceptionModel(nn.Module):
    def __init__(self, learner_modules, classifier, aux_blocks=[tuple]):
        super().__init__()
        self.classifier = classifier
        pre_aux1_blocks_pos = aux_blocks[0][1]
        incep = learner_modules[:pre_aux1_blocks_pos]
        #self.incep_grp1 = learner_modules[:pre_aux1_blocks_pos]
        self.incep_grp1 = nn.Sequential(*incep)
        self.aux1 = aux_blocks[0][0]
        
        pre_aux2_blocks_pos = aux_blocks[1][1]
        incep2 = learner_modules[pre_aux1_blocks_pos:pre_aux2_blocks_pos]
        self.incep_grp2 = nn.Sequential(*incep2)
        self.aux2 = aux_blocks[1][0]
        
        incep_grpblk = learner_modules[pre_aux2_blocks_pos:]
        self.incep_grp3 = nn.Sequential(*incep_grpblk)
        
        
    def forward(self, x):
        x = self.incep_grp1(x)
        aux_logit_1 = self.aux1(x)
        x = self.incep_grp2(x)
        aux_logit_2 = self.aux2(x)
        x = self.incep_grp3(x)
        #x = incepgrp5(x)
        logit = self.classifier(x)
        return {"aux_logits": [aux_logit_1, aux_logit_2],
                "main_logit": logit
                }
        

# class Inceptionv1Group(nn.Module):
#     def __init__(self, blocks, num_classes):
#         super().__init__()
        
#         self._aux = []
#         self.incepblk = []
#         for block in blocks:
#             if block is None:
#                 aux = InceptionAuxiliaryClassifier(num_classes=num_classes)
#                 self._aux.append(aux)
#             else:
#                 blk = InceptionBlock(f1x1=block[0], 
#                                     f3x3=block[1], 
#                                     f5x5=block[2],
#                                     fpool=block[3]
#                                     )
#                 self.incepblk.append(blk)
#         self.incep = nn.Sequential(**self.incepblk)
        
#         if self._aux is not None:
#             self.aux_classifier = nn.Sequential(**self._aux)
                
#     def forward(self, x):
#         x = self.incep(x)
#         if self._aux is not None:
#             aux_logits = self.aux_classifier(x)
#             return x, aux_logits
#         return x, None
                
                
        
def group(blocks, pooling: bool = True):
    incepblk = []
    for block in blocks:
        blk = InceptionBlock(f1x1=block[0], 
                            f3x3=block[1], 
                            f5x5=block[2],
                            fpool=block[3]
                            )
        incepblk.append(blk)
        
    if pooling:
        incepblk.append(nn.ZeroPad2d(padding=1))
        incepblk.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
    return nn.Sequential(*incepblk)
        

def group_auxiliary_classifiers(num_classes, group_pos=[2,3]):
    aux_blocks = []
    for grpos in group_pos:
        aux = InceptionAuxiliaryClassifier(num_classes=num_classes)
        aux_blocks.append((aux, grpos))
        
    return aux_blocks
      

def create_learner_inception_groups(group_configs=[{}]):
    learner_grps = []
    
    for grp_config in group_configs:
        grp_blks = grp_config.get("blocks")
        pooling = grp_config.get("pooling")
        grp_learner = group(blocks=grp_blks, pooling=pooling)
        learner_grps.append(grp_learner)
    
    return learner_grps
    

if __name__ == "__main__":
    preaux1_group = {"blocks": [((64,),  (96,128),   (16, 32), (32,)),  # 3a
                                ((128,), (128, 192), (32, 96), (64,)), # 3b
                                ((192,),  (96, 208), (16, 48), (64,)), # 4a
                                ],
                     "pooling": True
                     }
    
    group4_param_4a = [
                     None,] 				 # auxiliary classifier
    
    preaux2_group = {"blocks": [ ((160,), (112, 224), (24, 64), (64,)), # 4b
                                ((128,), (128, 256), (24, 64), (64,)), # 4c
                                ((112,), (144, 288), (32, 64), (64,)), # 4d
                                ],
                     "pooling": True
                     }  
    None    # auxiliary classifier
    postaux2_group1 =  {"blocks": [ ((256,), (160, 320), (32, 128), (128,)) # 4e
                                    ],
                        "pooling": True
                        } 
    
    postaux2_group2 = {"blocks":  [((256,), (160, 320), (32, 128), (128,)), # 5a
                                    ((384,), (192, 384), (48, 128), (128,)) # 5b
                                    ],
                       "pooling": False
                       }
    
    group_configs = [preaux1_group, preaux2_group, postaux2_group1, postaux2_group2]
    
    
    learner_modules = create_learner_inception_groups(group_configs=group_configs)
    aux_classifiers = group_auxiliary_classifiers(num_classes=5, group_pos=[2,3])
    classifier = InceptionClassifier(num_classes=5)
    
    stem = InceptionStem()
    learner_aux_task = InceptionModel(learner_modules=learner_modules, classifier=classifier,
                                      aux_blocks=aux_classifiers,
                                      )
    
    model = nn.Sequential(stem, learner_aux_task)
    #if example_input is None:
    #    if not in_channels:
    in_channels = 3
    example_input = torch.randn(1, in_channels, 224, 224)
    
    _ = model(example_input)
    model.apply(kernel_initializer)
    
    
    #LearnerInception_v1(group_params=group_params, num_classes=5)






# %%
