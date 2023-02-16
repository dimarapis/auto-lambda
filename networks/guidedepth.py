import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

""" Implementation of the DDRNet-23-slim Network, adopted from:
https://github.com/ydhongHIT/DDRNet/blob/main/segmentation/DDRNet_23_slim.py """

#BatchNorm2d = nn.SyncBatchNorm
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out

class DualResNet(nn.Module):
    def __init__(self, block, layers, out_features=19, planes=64, spp_planes=128, head_planes=128, augment=False, skip_out=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment
        self.skip_out = skip_out

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        """
        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, out_features)
        """
        self.final_layer = segmenthead(planes * 4, head_planes, out_features)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)
        if self.skip_out:
            x1 = x

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.final_layer(x + x_)
        return x_


def DualResNet_Backbone(pretrained=False, features=64):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], out_features=features,
                       planes=32, spp_planes=128, head_planes=64, augment=False)
    if pretrained:
        checkpoint = torch.load('./weights/' + "DDRNet23s_imagenet.pth",
                                map_location='cpu')

        model.load_state_dict(checkpoint, strict = False)
    return model

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode


    def forward(self, x):
        return F.interpolate(x, self.scale_factor, mode=self.mode)


class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2,3]) # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)



class Guided_Upsampling_Block(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(Guided_Upsampling_Block, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),  
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)


    def forward(self, guide, depth):
        x = self.feature_conv(depth)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            xy = torch.cat([x, y], dim=1)
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + depth)


""" Define GuideDepth """

class GuideDepth(nn.Module):
    def __init__(self,
            tasks, 
            pretrained=False,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()

        self.tasks = tasks

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        # define task specific layers
        if all (k in tasks for k in ('seg', 'depth', 'normal')):           
            self.pred_task1 = self.conv_layer([up_features[2], 1], pred=True)
            self.pred_task2 = self.conv_layer([up_features[2], 13], pred=True)
            self.pred_task3 = self.conv_layer([up_features[2], 3], pred=True)      
            self.decoders = nn.ModuleList([self.pred_task1, self.pred_task2, self.pred_task3])
      
        else:
            for k in tasks:
                if k == 'depth':
                    self.pred_task1 = self.conv_layer([up_features[2], 1], pred=True)  
                    self.decoders = nn.ModuleList([self.pred_task1])
                elif k == 'seg':
                    self.pred_task2 = self.conv_layer([up_features[2], 13], pred=True)  
                    self.decoders = nn.ModuleList([self.pred_task2])
                    
                elif k == 'normal':
                    self.pred_task3 = self.conv_layer([up_features[2], 3], pred=True)  
                    self.decoders = nn.ModuleList([self.pred_task3])

                else:
                    raise Exception("Messed up")
            

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block


    def forward(self, x):
        _, _, im_h, im_w = x.shape

        y = self.feature_extractor(x)

        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#, align_corners=True)
        y = self.up_3(x, y)

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](y), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)

        return out