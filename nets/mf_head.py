import torch
from torch import nn
from nets.conv_ import DSConv, DilationConv
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            DSConv(low_channels,out_channels,3,dilation = 2,padding = 2,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_high = nn.Sequential(
            DSConv(high_channels,out_channels,1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.cat_conv = nn.Sequential(
            DSConv(out_channels*2,out_channels,5,dilation = 2,padding = 4,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


        self.sig = nn.Sequential(
            DSConv(out_channels,out_channels,3,padding = 1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.ff_out = DSConv(out_channels, out_channels, 1, bias = False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)

        x_cat = torch.cat([x_low,x_high],dim = 1)
        ga = self.cat_conv(x_cat)
        # ga = self.sig(x)

        x_low = torch.mul(ga, x_low)
        x_high = torch.mul((1 - ga), x_high)

        x = self.ff_out(x_low + x_high)

        return x

class ASPP(nn.Module):
    def __init__(self, in_channel=768, depth=768):
        super(ASPP, self).__init__()
        self.pool2d = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, depth, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(depth * 5, depth, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(0.1)


    def forward(self, x):
        size = x.shape[2:]

        image_features = self.pool2d(x)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.block1(x)
        atrous_block6 = self.block2(x)
        atrous_block12 = self.block3(x)
        atrous_block18 = self.block4(x)
        concat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        block5 = self.block5(concat)
        net = self.dropout(block5)
        return net

class MF_Head(nn.Module):
    '''
    Multiscale feature fusion
    '''
    def __init__(self,in_channels=[32, 64, 160, 256],num_classes=2):
        super(MF_Head,self).__init__()

        self.ff0 = FeatureFusion(in_channels[3],in_channels[2],256)
        self.ff1 = FeatureFusion(256,in_channels[1],256)
        self.ff2 = FeatureFusion(256,in_channels[0],256)
        self.aspp = ASPP(in_channel=768, depth=768)
        self.seg = nn.Conv2d(768,num_classes,kernel_size = 1)

    def forward(self,inputs):
        c1, c2, c3, c4 = inputs
        ff0_out = self.ff0(c4,c3)
        ff1_out = self.ff1(ff0_out,c2)
        ff2_out = self.ff2(ff1_out,c1)

        ff0_out = F.interpolate(ff0_out,ff2_out.shape[2:],mode='bilinear', align_corners=True)
        ff1_out = F.interpolate(ff1_out, ff2_out.shape[2:], mode = 'bilinear', align_corners = True)
        x = torch.cat([ff0_out,ff1_out,ff2_out],dim = 1)

        aspp_out = self.aspp(x)
        out = self.seg(aspp_out)

        return out

