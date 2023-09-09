import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNeXt101_32X8D_Weights

class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.base = models.resnext50_32x4d(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=16)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=8)
        self.fc = nn.Conv2d(64, 11, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.base(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        kpts = self.fc(x)
        return kpts

class DepthmapModel(nn.Module):
    def __init__(self):
        super(DepthmapModel, self).__init__()
        self.base = models.resnext50_32x4d(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=16)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=8)
        self.fc = nn.Conv2d(64, 11, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.base(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        depth = self.fc(x)
        return depth



class KeypointDepthmapModel(nn.Module):
    def __init__(self):
        super(KeypointDepthmapModel, self).__init__()
        self.keypoint_model = KeypointModel()
        self.depthmap_model = DepthmapModel()

    def forward(self, x):
        kpts = self.keypoint_model(x)
        depth = self.depthmap_model(x)
        heads = {'hm_c': kpts, 'depth': depth}
        return heads
