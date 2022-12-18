import torch.nn as nn
import torch

from model.HRNetv2 import HRNetv2
from model.HRNetSmall import HRNetSmall


class CPHRNetv2(nn.Module):
    def __init__(self, c1, c2, k):
        super(CPHRNetv2, self).__init__()
        self.k = k
        self.features_extractor = HRNetv2(c1)

        self.conv1_stage1 = nn.Sequential(
            nn.Conv2d(15 * c1, 15 * c1, kernel_size=1, padding=0),
            nn.BatchNorm2d(15 * c1),
            nn.ReLU(),
            nn.Conv2d(15 * c1, self.k, kernel_size=1, padding=0),
        )

        self.bridge1 = nn.Sequential(
            nn.Conv2d(15 * c1, 15 * c1, kernel_size=5, padding=2),
            nn.BatchNorm2d(15 * c1),
            nn.ReLU()
        )

        self.conv1_stage2 = HRNetSmall(15 * c1 + self.k, c2)
        self.conv2_stage2 = nn.Sequential(
            nn.Conv2d(7 * c2, 7 * c2, kernel_size=1, padding=0),
            nn.BatchNorm2d(7 * c2),
            nn.ReLU(),
            nn.Conv2d(7 * c2, self.k, kernel_size=1, padding=0),
        )

        self.bridge2 = nn.Sequential(
            nn.Conv2d(15 * c1, 15 * c1, kernel_size=5, padding=2),
            nn.BatchNorm2d(15 * c1),
            nn.ReLU()
        )

        self.conv1_stage3 = HRNetSmall(15 * c1 + self.k, c2)
        self.conv2_stage3 = nn.Sequential(
            nn.Conv2d(7 * c2, 7 * c2, kernel_size=1, padding=0),
            nn.BatchNorm2d(7 * c2),
            nn.ReLU(),
            nn.Conv2d(7 * c2, self.k, kernel_size=1, padding=0),
        )

    def stage1(self, image):
        x = self.conv1_stage1(image)
        return x

    def stage2(self, features1, heatmap):
        features1 = self.bridge1(features1)
        x = torch.cat([features1, heatmap], dim=1)
        x = self.conv1_stage2(x)
        x = self.conv2_stage2(x)
        return x

    def stage3(self, features1, heatmap):
        features1 = self.bridge2(features1)
        x = torch.cat([features1, heatmap], dim=1)
        x = self.conv1_stage3(x)
        x = self.conv2_stage3(x)
        return x


    def forward(self, image):
        features1 = self.features_extractor(image)

        heatmap1 = self.stage1(features1)

        heatmap2 = self.stage2(features1, heatmap1)

        heatmap3 = self.stage3(features1, heatmap2)

        return heatmap1, heatmap2, heatmap3
