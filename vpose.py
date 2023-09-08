from __future__ import division
from __future__ import print_function
import os
import logging
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

import torch.nn as nn
import torchvision.models as models

class BottleneckX(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=32,
                 base_width=4, dilation=1, norm_layer=None):
        super(BottleneckX, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups,
                               dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * 4),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PoseDepthResNet(nn.Module):
    def __init__(self, block=BottleneckX, layers=[3, 4, 23, 3], layers_in=3):
        super(PoseDepthResNet, self).__init__()
        
        # Base ResNeXt101_32x8d model
        base_model = models.resnext101_32x8d(pretrained=True)

        # Remove last two layers (avgpool and fc) from base model
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        # Replace _make_layer calls with BottleneckX calls
        self.layer1 = nn.Sequential(
            base_model[0],
            base_model[1],
            base_model[2],
            base_model[3]
        )
        
        self.layer2 = nn.Sequential(
            BottleneckX(256 * block.expansion, 128),
            *[BottleneckX(512 * block.expansion, 128) for _ in range(layers[0] - 1)]
        )
        
        self.layer3 = nn.Sequential(
            BottleneckX(1024 * block.expansion, 256),
            *[BottleneckX(1024 * block.expansion, 256) for _ in range(layers[1] - 1)]
        )
        
        self.layer4 = nn.Sequential(
            BottleneckX(2048 * block.expansion, 512),
            *[BottleneckX(2048 * block.expansion, 512) for _ in range(layers[2] - 1)]
        )

        # Output layers
        self.depth_layer = nn.Conv2d(256,
                                     11,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        
        self.keypoint_layer = nn.Conv2d(256,
                                        11,
                                        kernel_size=1,
                                        stride=1)

    def forward(self, x):
        
         # Input layer
         x = x.float()
         x /= 255.0
        
         # ResNeXt Layers
         x = self.layer1(x)
         x = self.layer2(x)
         x = self.layer3(x)
         x = self.layer4(x)

         depth = self.depth_layer(x)
         kpts = self.keypoint_layer(x)
         heads = {'hm_c': kpts.view(kpts.size(0), -1, 1), 'depth': depth.view(depth.size(0), -1)}

         return heads


resnet_spec = {101x: (BottleneckX, [3, 4, 23, 3])}


def get_pose_depth_net(num_layers, train = True, concat=False,**kwargs):

    block_class, layers = resnet_spec[num_layers]

    if concat:
        model = PoseDepthResNet(block_class, layers, layers_in=22, **kwargs)

    else:
        model = PoseDepthResNet(block_class, layers, **kwargs)

    if train:

        if num_layers == 101x:
           state_dict = from torchvision.models.resnet import ResNeXt101_32X8D_Weights            

    return model




