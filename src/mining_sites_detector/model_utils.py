import torch
import torch.nn as nn
from .densenet import ClassifierDenseNet

from .models.resnet import Resnet_stem
             
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
    elif stem_type == "inception_v1":
        return InceptionStem(out_channels=out_channels)
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
    elif stem_type == "inception_v1":
        learner_module, aux_classifiers = learner_inception_v1(x=None,
                                                                n_classes=num_classes,
                                                                group_params=group_params
                                                                )
        task_module = InceptionClassifier(num_classes=num_classes)

    model = nn.Sequential(stem_module, learner_module,task_module)
    if example_input is None:
        if not in_channels:
            in_channels = 3
        example_input = torch.randn(1, in_channels, 224, 224)
    
    _ = model(example_input)
    model.apply(kernel_initializer)
    return model
