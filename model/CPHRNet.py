import torch.nn as nn
import torch.nn.functional as F
import torch

from model.HRNetv2 import HRNetv2


class CPHRNet(nn.Module):
    def __init__(self, k, c):
        super(CPHRNet, self).__init__()
        self.k = k
        self.c = c

        self.features_extractor = HRNetv2(self.c)

        self.conv1_stage1 = nn.Conv2d(self.c * 15, 128, kernel_size=3, padding=1)
        self.Mconv1_stage1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv2_stage1 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv4_stage2 = nn.Conv2d(self.c * 15, 32, kernel_size=3, padding=1)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.k, 128, kernel_size=3, padding=1)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(self.c * 15, 32, kernel_size=3, padding=1)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.k, 128, kernel_size=3, padding=1)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(self.c * 15, 32, kernel_size=3, padding=1)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.k, 128, kernel_size=3, padding=1)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(self.c * 15, 32, kernel_size=3, padding=1)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.k, 128, kernel_size=3, padding=1)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(self.c * 15, 32, kernel_size=3, padding=1)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.k, 128, kernel_size=3, padding=1)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

    def _stage1(self, image):

        x = F.relu(self.conv1_stage1(image))
        x = F.relu(self.Mconv1_stage1(x))
        x = self.Mconv2_stage1(x)

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map):

        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map):

        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map):

        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map):

        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map):
        
        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)

        return x

    def forward(self, image):
        features = self.features_extractor(image)
        
        conv7_stage1_map = self._stage1(features)
        Mconv5_stage2_map = self._stage2(features, conv7_stage1_map)
        Mconv5_stage3_map = self._stage3(features, Mconv5_stage2_map)
        Mconv5_stage4_map = self._stage4(features, Mconv5_stage3_map)
        Mconv5_stage5_map = self._stage5(features, Mconv5_stage4_map)
        Mconv5_stage6_map = self._stage6(features, Mconv5_stage5_map)

        return conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map
