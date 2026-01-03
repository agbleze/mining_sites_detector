#%% 
import numpy as np
import torch
from torch_snippets import Report
import torch.nn as nn
import os
from copy import deepcopy

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5).squeeze() 
    is_correct = (is_correct == y)
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    prediction = prediction.squeeze()
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()


def train_batch(x, y, model, loss_fn, optimizer):
    optimizer.zero_grad()
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction.squeeze(), 
                         y)
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()


def trigger_training_process(train_dataload, val_dataload, model, loss_fn,
                             optimizer, num_epochs: int, device="cuda",
                             model_store_dir="model_store", 
                             model_name="mining_site_detector_model",
                             checkpoint_interval: int = 1,
                             train_on_indice = False,
                             ):
    os.makedirs(model_store_dir, exist_ok=True)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    log = Report(n_epochs=num_epochs)
    
    for epoch in range(num_epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies, val_epoch_losses = [], []
        for ix, batch in enumerate(iter(train_dataload)):
            x = batch["indice"].to(device) if train_on_indice else batch["image"].to(device) 
            y = batch["label"].to(device)
            model.to(device)
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        
        for ix, batch in enumerate(iter(train_dataload)):
            x = batch["indice"].to(device) if train_on_indice else batch["image"].to(device)
            y = batch["label"].to(device)
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        
        for ix, batch in enumerate(iter(val_dataload)):
            x, y = batch["image"].to(device), batch["label"].to(device)
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
            val_batch_loss = val_loss(x, y, model, loss_fn=loss_fn)
            val_epoch_losses.append(val_batch_loss)
            
        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        
        #if (epoch + 1) % 1 == 0:
        log.record(pos=epoch+1, trn_loss=train_epoch_loss,
                   trn_acc=train_epoch_accuracy,
                   val_acc=val_epoch_accuracy,
                   val_loss=val_epoch_loss,
                   end="\r"
                   )
        log.report_avgs(epoch+1)
        
        if (epoch +1) % checkpoint_interval == 0:
            model_path = os.path.join(model_store_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save(deepcopy(model.to("cpu").state_dict()), model_path)
            
            # save model in state for infernece / resuming training
            print("saving model as checkpoint")
            resume_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_resumable_epoch_{epoch+1}.pth'
                                             )
            torch.save({"epoch": epoch+1,
                        "model_state_dict": deepcopy(model.to("cpu").state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "loss": deepcopy(val_epoch_loss),
                        },
                       resume_model_path
                       )
            
            # save model as torchscript file for easy loading
            print("Exporting to torchscript")
            torchscript_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_torchscript_epoch_{epoch+1}.pt'
                                             )
            model_scripted = torch.jit.script(deepcopy(model.to("cpu")))
            model_scripted.save(torchscript_model_path)       
        
    return {"train_loss": train_losses,
            "train_accuracy": train_accuracies,
            "valid_loss": val_losses,
            "valid_accuracy": val_accuracies
            }



device = "cuda" if torch.cuda.is_available() else "cpu"
def conv_layer(in_chan, out_chan, kernel_size, stride=1):
    return nn.Sequential(
                    nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                              kernel_size=kernel_size, 
                              stride=stride
                              ),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=out_chan),
                    nn.MaxPool2d(kernel_size=2)
                )

def get_model():
    model = nn.Sequential(conv_layer(12, 64, 3),
                          conv_layer(64, 512, 3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(512, 512, 3),
                          nn.Flatten(),
                          nn.Linear(18432, 1),
                          #nn.Sigmoid()
                          ).to(device)
    
    #loss_fn = nn.BCELoss().to(device=device)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer



### ARCH Adaptation

"""
Let indices be created during data loading in NonGeoMineSiteClassificationDataset 

paricularly in _load_image, then return it dict with key indice

Arch innovations

#####  Branching options  ######
Encoder A branch 3 channels--- Bands Red (4), Green (3), Blue (2)
Encoder B branch --- Rest of 9 bands

## Branching By information content (for compact multi‑branch models)

Branch A: RGB (3 channels)
Texture, edges, shapes.

Branch B: Everything else (9 channels)
All the spectral richness.


## Branching By spatial resolution 

Branch A - 10m bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
Branch B - 20m bands: B5, B6, B7 (Red Edge), B8A (NIR narrow), B11, B12 (SWIR)
Branch C - 60m bands: B1 (Coastal), B9 (Water vapor), B10 (Cirrus)


## Branching By spectral family (for semantic separation)
This is great if you want the model to learn different physical processes.

Branch A: RGB (3 channels)
Human‑interpretable structure.

Branch B: Red Edge + NIR (4 channels)
Vegetation health, chlorophyll, stress.

Branch C: SWIR (2 channels)
Moisture, soil, burn severity.

Branch D: Atmospheric bands (1–2 channels)
Clouds, aerosols.

Representation fusion options
1. Concatenation
Encoder A output concatenated with Encoder B output

2. Addition
Encoder A output added to Encoder B output 
- Encoder A and B must project to same channel dimension

3. Gating
- Project encoder A and B embeddings to same dimension
- Compute similarity score (cosine similarity) between two embeddings
- Pass similarity score through small MLP with sigmoid activation to get gate value
- Applying the gate to weight encoder A and (1-gate) to weight encoder B
- Fused representation passed to task head for prediction




##################  Designing the architecture  ##################
# Separate Task branch that will remain largely unchanged

# Learner branches that will be modified to accept different inputs


# Fusion mechanism to combine the outputs of the learner branches
"""



class ResneXt_stem(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=7, strides=2):
        super(ResneXt_stem, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=strides,
                                bias=False, padding_mode='same'
                            )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, strides=2, padding_mode='same')
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x
    


class Resnet_stem(nn.Module):
    def __init__(self, out_channels=64, kernel_size=7, strides=2):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels,
                                kernel_size=kernel_size, stride=strides, 
                                bias=False
                            )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.get_zero_padded(padding=3)(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.get_zero_padded(padding=1)(x)
        x = self.pool(x)
        return x
        
    def get_zero_padded(self, padding):
        
        return nn.ZeroPad2d(padding=padding)


class Xception_stem(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, strides=2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.act = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x
        
        
#### learner ##########

def learner(inputs, groups):
    outputs = inputs
    for group_params in groups:
        outputs = group(outputs, **group_params)   
    return outputs

def group(inputs, **blocks):
    outputs = inputs
    for block_params in blocks:
        outputs = block(**block_params)
    return outputs




class IdentityBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        # 1 x 1 residual block
        #out_channels = out_channels // 4
        self.conv1 = nn.LazyConv2d(out_channels=out_channels,
                                    kernel_size=(1,1), stride=(1,1), 
                                    bias=False,
                                    )
        self.bn1 = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        
        # 3 x 3 residual block
        # dimensionality reduction - likely that out_channels < in_channels by 4x
        self.conv2 = nn.LazyConv2d(out_channels=out_channels,
                                    kernel_size=(3,3), stride=(1,1), 
                                    bias=False, padding=1
                                    )
        self.bn2 = nn.LazyBatchNorm2d()
        #self.act2 = nn.ReLU()
        
        # 1 x 1 residual block
        # Dimensionality restoration - out_channels * 4
        self.conv3 = nn.LazyConv2d(out_channels=out_channels * 4,
                                    kernel_size=(1,1), stride=(1,1), bias=False
                                    )
        self.bn3 = nn.LazyBatchNorm2d()
        
        # conv_layers = [self.conv1, self.conv2, self.conv3]
        # for conv_layer in conv_layers:
        #     kernel_initializer(conv_layer)
            
    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += shortcut
        x = nn.ReLU()(x)
        return x
        
        
class ProjectionBlock(nn.Module):
    def __init__(self, out_channels, strides,
                 **metaparameters
                 ):
        super().__init__()
        
        # Projection shortcut with 4x increased filters to macth output block for addition
        self.shortcut = nn.LazyConv2d(out_channels=out_channels * 4,
                                  kernel_size=(1,1), stride=strides,
                                  bias=False
                                  )
        self.shortcut_bn = nn.LazyBatchNorm2d() #nn.BatchNorm2d(num_features=out_channels * 4)
        
        # 1 x 1 residual block
        # dimensionality reduction and feature pooling with 2 x2 stride
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        #                        kernel_size=(1,1), strides=strides,
        #                        bias=False,
        #                        )
        self.conv1 = nn.LazyConv2d(out_channels=out_channels,
                                kernel_size=(1,1), stride=strides, bias=False
                                )
        #self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn1 = nn.LazyBatchNorm2d()
        self.act1 = nn.ReLU()
        
        # 3 x 3 residual block
        self.conv2 = nn.LazyConv2d(out_channels=out_channels,
                                    kernel_size=(3,3), stride=(1,1), bias=False,
                                    padding=1
                                    )
        #self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.LazyBatchNorm2d()

        # 1 x 1 residual block
        # Dimensionality restoration - out_channels * 4
        # self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4,
        #                        kernel_size=(1,1), stride=(1,1), bias=False
        #                        )
        self.conv3 = nn.LazyConv2d(out_channels=out_channels * 4,
                                    kernel_size=(1,1), stride=(1,1), bias=False
                                    )
        #self.bn3 = nn.BatchNorm2d(num_features=out_channels * 4)
        self.bn3 = nn.LazyBatchNorm2d()
        self.act2 = nn.ReLU()
        
        self.act3 = nn.ReLU()
        
        # conv_layers = [self.conv1, self.conv2, self.conv3, self.shortcut]
        # for conv_layer in conv_layers:
        #     kernel_initializer(conv_layer)
            
    def forward(self, x):
        shortcut = self.shortcut(x)
        shortcut = self.shortcut_bn(shortcut)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += shortcut
        x = self.act3(x)
        return x


def group(n_filters, n_blocks, strides=(2,2)):
    proj = ProjectionBlock(out_channels=n_filters, strides=strides)
    
    id_block_collection = []
    for _ in range(n_blocks):
        id_block = IdentityBlock(out_channels=n_filters) 
        id_block_collection.append(id_block)
        
    return nn.Sequential(proj, *id_block_collection)


def learner(groups):
    # First group with no downsampling
    n_filters, n_blocks = groups.pop(0)
    grp1 = group(n_filters=n_filters, n_blocks=n_blocks, strides=(1,1))
    
    # Remaining groups with downsampling - strided
    other_gpr_collections = []
    for n_filters, n_blocks in groups:
        grp = group(n_filters=n_filters, n_blocks=n_blocks, strides=(2,2))
        other_gpr_collections.append(grp)
        
    return nn.Sequential(grp1, *other_gpr_collections)


class ClassifierResnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.act = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.act(x)
        return x


###

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
        


class SqueezeNetStem(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=out_channels,
                                   kernel_size=7, stride=2,
                                   padding=3,
                                   )
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.maxpool(x)
        return x
    
    


class ClassifierSqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()   
        self.conv1 = nn.LazyConv2d(kernel_size=1, out_channels=num_classes)
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.relu = nn.ReLU()
        self.act = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.globalavgpool(x)
        x = self.act(x)
        return x
        

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
        
            
    
           
def kernel_initializer(m, kernel_initializer="he_normal"):
    if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
            


def get_stem_module(stem_type, in_channels=None, out_channels=64):
    if stem_type == "resnet":
        return Resnet_stem(out_channels=out_channels)
    elif stem_type == "resnext":
        return ResneXt_stem(in_channels=in_channels)
    elif stem_type == "xception":
        return Xception_stem(in_channel=in_channels)
    elif stem_type == "densenet":
        return StemDenseNet(out_channels=out_channels)
    else:
        raise ValueError(f"Unsupported stem type: {stem_type}")
    
    
def build_model(stem_type, num_classes, group_params,
                example_input=None, in_channels=None,
                n_filters: int = 32,
                reduction: float = 0.5
                ):
    stem_module = get_stem_module(stem_type=stem_type,
                                in_channels=in_channels,
                                out_channels=64
                                )
    print(f"Using stem: {stem_type}")
    if stem_type == "resnet":
        learner_module = learner(groups=group_params)
        task_module = ClassifierResnet(num_classes=num_classes)
    elif stem_type == "densenet":
        learner_module = learner_densenet(groups=group_params,
                                          n_filters=n_filters,
                                          reduction=reduction
                                          )
        task_module = ClassifierDenseNet(num_classes=num_classes)

    model = nn.Sequential(stem_module, learner_module,task_module)
    if example_input is None:
        if not in_channels:
            in_channels = 3
        example_input = torch.randn(1, in_channels, 224, 224)
    
    _ = model(example_input)
    model.apply(kernel_initializer)
    return model

#%%


resnet_model = build_model(stem_type="resnet",
            num_classes=2,
            group_params=[(64, 3), (128, 4), (256, 6), (512, 3)],
            in_channels=3
            )
#%%
resnet_model

#%%
densenet_model = build_model(stem_type="densenet",
                             num_classes=2,
                             group_params=[6,12,24,16],#n_blocks per group
                             n_filters=32,
                             reduction=0.5,
                             in_channels=3
                             )

#%%
densenet_model.parameters()

#%%            
"""

convolution output size formula:

output_size = ((in_size + 2 * padding - kernel_size) / stride) + 1
"""


#%%

import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Create the block
block = MyBlock()

# Create random input: batch=1, channels=3, height=9, width=9
x = torch.randn(1, 3, 9, 9)

# Run it
y = block(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

# %%
import torch
import torch.nn as nn

conv = nn.Conv2d(3, 8, kernel_size=3, padding="same")
x = torch.randn(1, 3, 9, 9)
y = conv(x)

print(x.shape)
print(y.shape)
# %%
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(64, 3, padding=1)
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2,3])
        x = self.fc(x)
        return x

def kernel_initializer(m):
    if isinstance(m, (nn.LazyConv2d, nn.LazyLinear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = Block()

# Lazy layers not initialized yet
#print("Before forward:", model.conv.weight.shape)

# Trigger initialization
dummy = torch.randn(1, 3, 32, 32)
_ = model(dummy)

print("After forward:", model.conv.weight.shape)

# Now apply init
model.apply(kernel_initializer)

# %%
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.gap(x)          # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.fc(x)           # (B, n_classes)
        return torch.softmax(x, dim=1)

# ---- Test it ----

model = Classifier(n_classes=10)

# Random input: batch=4, channels=64, H=7, W=7
x = torch.randn(4, 64, 7, 7)

# First forward pass initializes LazyLinear
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Output (first row):", y[0])
# %%
import torch
import torch.nn as nn

# # --- Stem module (corrected version) ---
# class SqueezeNetStem(nn.Module):
#     def __init__(self, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=None,
#             out_channels=out_channels,
#             kernel_size=7,
#             stride=2,
#             padding=3
#         )
#         nn.init.xavier_uniform_(self.conv1.weight)
#         if self.conv1.bias is not None:
#             nn.init.zeros_(self.conv1.bias)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.maxpool(x)
#         return x

#%% --- Test ---
if __name__ == "__main__":
    #%%
    model = SqueezeNetStem(out_channels=96)

    # Fake input: batch=1, channels=3, height=224, width=224
    x = torch.randn(1, 3, 224, 224)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

# %%
