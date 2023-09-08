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

class BottleneckX(nn.Module):
    def __init__(self):
        super(BottleneckX, self).__init__()
        self.base = models.resnext101_32x8d(pretrained=True)
        self.base = nn.Sequential(*list(self.base.children())[:-2])

    def forward(self, x):
        out = self.base(x)
        return out


class PoseDepthResNet(nn.Module):
    def __init__(self, block, layers, layers_in=3, **kwargs):
        self.inplanes = 64
        super(PoseDepthResNet, self).__init__()
        self.conv1 = nn.Conv2d(layers_in, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = block(128 * block.expansion, 128, stride=2) # Use BottleneckX instead of _make_layer
        self.layer3 = block(256 * block.expansion, 256, stride=2) # Use BottleneckX instead of _make_layer
        self.layer4 = block(512 * block.expansion, 512, stride=2) # Use BottleneckX instead of _make_layer
        self.depth_layer = nn.Conv2d(64, 11, kernel_size=1, stride=1,
                                     padding=0)
        self.keypoint_layer = nn.Conv2d(64, 11 * 4, kernel_size=1,
                                        stride=1)

    def forward(self, x):
        # Input layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth = self.depth_layer(x)
        kpts = self.keypoint_layer(x)
        heads = {'hm_c': kpts.view(kpts.size(0), -1, 4), 'depth': depth.view(depth.size(0), -1)}

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




