

import torch
import torch.nn as nn


class InceptionV3Stem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_a = nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding="valid", bias=False)
        self.bn = nn.BatchNorm2d()
        self.act = nn.ReLU()
        
        self.conv_b = nn.LazyConv2d(out_channels=32, kernel_size=3, stride=1, padding="valid", bias=False)
        self.conv_c = nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding="same", bias=False)
        self.conv_d = nn.LazyConv2d(out_channels=80, kernel_size=1, stride=1, padding="valid", bias=False)
        self.conv_e = nn.LazyConv2d(out_channels=192, kernel_size=3, stride=1, padding="valid", bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
    def forward(self, x):
        # coarse filter of v1 (7x7) factorized into 3x3
        x = self.conv_a(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv_b(x)
        x = self.bn(x)
        x = self.act(x)
        
        # third 3x3. filters are doubled and paddling added
        x = self.conv_c(x)
        x = self.bn(x)
        x = self.act(x)
        
        # pooled feature maps will be reduced by 75%
        x = self.maxpool(x)
        
        # 3x3 reduction
        x = self.conv_d(x)
        x = self.bn(x)
        x = self.act(x)
        
        # Dimensionality expansion
        x = self.conv_e(x)
        x = self.bn(x)
        x = self.act(x)
        
        # pooled feature maps reduce by 75%
        x = self.maxpool(x)
        return x
        
    


class InceptionV3BlockA(nn.Module):
    def __init__(self, f1x1, f3x3, f5x5, fpool):
        super().__init__()
        
        self.act = nn.ReLU()
        self.zeropad1 = nn.ZeroPad2d(1)
        #self.zeropad2 = nn.ZeroPad2d(2)
        self.bn = nn.BatchNorm2d()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding="same")
        
        self.f1x1_conv = nn.LazyConv2d(out_channels=f1x1[0], kernel_size=1, stride=1, padding="same", bias=False)
        
        self.f3x3_conv1x1 = nn.LazyConv2d(out_channels=f3x3[0], kernel_size=1, padding="same", stride=1, bias=False)
        self.f3x3_conv3x3 = nn.LazyConv2d(out_channels=f3x3[1], kernel_size=3, stride=1, padding="same", bias=False)
        
        self.f5x5_conv1x1 = nn.LazyConv2d(out_channels=f5x5[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.f5x5_conv3x3_a = nn.LazyConv2d(out_channels=f5x5[1], kernel_size=3, stride=1, padding="same", bias=False)
        self.f5x5_conv3x3_b = nn.LazyConv2d(out_channels=f5x5[2], kernel_size=3, stride=1, padding="same", bias=False)
        
        self.pool_conv1x1 = nn.LazyConv2d(out_channels=fpool[0], kernel_size=1, stride=1, padding="same", bias=False)
        
        
    def forward(self, x):
        x_f1x1 = self.f1x1_conv(x)
        x_f1x1 = self.bn(x_f1x1)
        x_f1x1 = self.act(x_f1x1)

        x_f3x3 = self.f3x3_conv1x1(x)
        x_f3x3 = self.bn(x_f3x3)
        x_f3x3 = self.act(x_f3x3)
        #x_f3x3 = self.zeropad1(x_f3x3)
        x_f3x3 = self.f3x3_conv3x3(x_f3x3)
        
        x_f5x5 = self.f5x5_conv1x1(x)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        x_f5x5 = self.f5x5_conv3x3_a(x_f5x5)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        x_f5x5 = self.f5x5_conv3x3_b(x_f5x5)
        x_f5x5 = self.bn(x_f5x5)
        x_f5x5 = self.act(x_f5x5)
        
        
        x_pool = self.avgpool(x)
        x_pool = self.pool_conv1x1(x_pool)
        x_pool = self.bn(x_pool)
        x_pool = self.act(x_pool)
        
        output = torch.concat([x_f1x1, x_f3x3, x_f5x5, x_pool], dim=1)
        return output
        
        
        
        
    
    
class InceptionV3BlockB(nn.Module):
    def __init__(self, f1x1, f1xn, f1xndbl, fpool):
        super().__init__()
        
        self.act = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding="same")
        self.bn = nn.BatchNorm2d()
        
        self.f1x1_conv = nn.LazyConv2d(out_channels=f1x1[0], kernel_size=1, stride=1, padding="same", bias=False)
        
        self.f1xn_conv1x1 = nn.LazyConv2d(out_channels=f1xn[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.f1xn_conv1xn = nn.LazyConv2d(out_channels=f1xn[1], kernel_size=(1,7), stride=1, padding="same", bias=False)
        self.f1xn_convnx1 = nn.LazyConv2d(out_channels=f1xn[2], kernel_size=(7, 1), stride=1, padding="same", bias=False)
        
        self.f1xndbl_conv1x1 = nn.LazyConv2d(out_channels=f1xndbl[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.f1xndbl_conv1xn_a = nn.LazyConv2d(out_channels=f1xndbl[1], kernel_size=(1,7), stride=1, padding="same", bias=False)
        self.f1xndbl_convnx1_a = nn.LazyConv2d(out_channels=f1xndbl[2], kernel_size=(7, 1), stride=1, padding="same", bias=False)
        self.f1xndbl_conv1xn_b = nn.LazyConv2d(out_channels=f1xndbl[3], kernel_size=(1,7), stride=1, padding="same", bias=False)
        self.f1xndbl_convnx1_b = nn.LazyConv2d(out_channels=f1xndbl[4], kernel_size=(7, 1), stride=1, padding="same", bias=False) 
        
        self.pool_conv1x1 = nn.LazyConv2d(out_channels=fpool[0], kernel_size=1, stride=1, padding="same", bias=False)
        
        
def forward(self, x):
    x_f1x1 = self.f1x1_conv(x)
    x_f1x1 = self.bn(x_f1x1)
    x_f1x1 = self.act(x_f1x1)
    
    x_f1xn = self.f1xn_conv1x1(x)
    x_f1xn = self.bn(x_f1xn)
    x_f1xn = self.act(x_f1xn)
    x_f1xn = self.f1xn_conv1xn(x_f1xn)
    x_f1xn = self.bn(x_f1xn)
    x_f1xn = self.act(x_f1xn)
    x_f1xn = self.f1xn_convnx1(x_f1xn)
    x_f1xn = self.bn(x_f1xn)
    x_f1xn = self.act(x_f1xn)
    
    x_f1xndbl = self.f1xndbl_conv1x1(x)
    x_f1xndbl = self.bn(x_f1xndbl)
    x_f1xndbl = self.act(x_f1xndbl)
    x_f1xndbl = self.f1xndbl_conv1xn_a(x_f1xndbl)
    x_f1xndbl = self.bn(x_f1xndbl)
    x_f1xndbl = self.act(x_f1xndbl)
    x_f1xndbl = self.f1xndbl_convnx1_a(x_f1xndbl)
    x_f1xndbl = self.bn(x_f1xndbl)
    x_f1xndbl = self.act(x_f1xndbl)
    x_f1xndbl = self.f1xndbl_conv1xn_b(x_f1xndbl)
    x_f1xndbl = self.bn(x_f1xndbl)
    x_f1xndbl = self.act(x_f1xndbl)
    x_f1xndbl = self.f1xndbl_convnx1_b(x_f1xndbl)
    x_f1xndbl = self.bn(x_f1xndbl)
    x_f1xndbl = self.act(x_f1xndbl)
    
    x_pool = self.avgpool(x)
    x_pool = self.pool_conv1x1(x_pool)
    x_pool = self.bn(x_pool)
    x_pool = self.act(x_pool)
    
    output = torch.concat([x_f1x1, x_f1xn, x_f1xndbl, x_pool], dim=1)
    return output    



class InceptionV3BlockC(nn.Module):
    def __init__(self, f1x1, f3x3, f3x3dbl, fpool):
        super().__init__()
        
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d()
        
        self.f1x1_conv1x1 = nn.LazyConv2d(out_channels=f1x1[0], kernel_size=1, stride=1, padding="same", bias=False)
        
        self.f3x3_conv1x1 = nn.LazyConv2d(out_channels=f3x3[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.f3x3_conv1x3 = nn.LazyConv2d(out_channels=f3x3[1], kernel_size=(1,3), stride=1, padding="same", bias=False)
        self.f3x3_conv3x1 = nn.LazyConv2d(out_channels=f3x3[2], kernel_size=(3,1), stride=1, padding="same", bias=False)
        
        self.f3x3dbl_conv1x1 = nn.LazyConv2d(out_channels=f3x3dbl[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.f3x3dbl_conv3x3 = nn.LazyConv2d(out_channels=f3x3dbl[1], kernel_size=3, stride=1, padding="same", bias=False)
        self.f3x3dbl_conv1x3 = nn.LazyConv2d(out_channels=f3x3dbl[2], kernel_size=(1 ,3), stride=1, padding="same", bias=False)
        self.f3x3dbl_conv3x1 = nn.LazyConv2d(out_channels=f3x3dbl[3], kernel_size=(3, 1), stride=1, padding="same", bias=False)
        
        self.fpool_conv1x1 = nn.LazyConv2d(out_channels=fpool[0], kernel_size=1, stride=1, padding="same", bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding="same")
        
        
    def forward(self, x):
        x_f1x1 = self.f1x1_conv1x1(x)
        x_f1x1 = self.bn(x_f1x1)
        x_f1x1 = self.act(x_f1x1)
        
        
        x_f3x3 = self.f3x3_conv1x1(x)
        x_f3x3 = self.bn(x_f3x3)
        x_f3x3 = self.act(x_f3x3)
        
        x_f3x3_1x3 = self.f3x3_conv1x3(x_f3x3)
        x_f3x3_1x3 = self.bn(x_f3x3_1x3)
        x_f3x3_1x3 = self.act(x_f3x3_1x3)
        
        x_f3x3_3x1 = self.f3x3_conv3x1(x_f3x3)
        x_f3x3_3x1 = self.bn(x_f3x3_3x1)
        x_f3x3_3x1 = self.act(x_f3x3_3x1)
        
        
        x_f3x3dbl = self.f3x3dbl_conv1x1(x)
        x_f3x3dbl = self.bn(x_f3x3dbl)
        x_f3x3dbl = self.act(x_f3x3dbl)
        
        x_f3x3dbl = self.f3x3dbl_conv3x3(x_f3x3dbl)
        x_f3x3dbl = self.bn(x_f3x3dbl)
        x_f3x3dbl = self.act(x_f3x3dbl)
        
        
        
        
        
        
        
        