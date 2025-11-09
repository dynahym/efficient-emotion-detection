import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_channels)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_channels)
        
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze_bn(self.squeeze(x)))
        
        expand1x1_out = self.expand1x1_bn(self.expand1x1(x))
        expand3x3_out = self.expand3x3_bn(self.expand3x3(x))
        
        return self.expand_activation(torch.cat([expand1x1_out, expand3x3_out], 1))


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SqueezeNet, self).__init__()
        
        # Initial convolution adapted for 48x48 input
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # Output: 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 12x12

            Fire(64, 16, 64, 64),      # 128 channels
            Fire(128, 16, 64, 64),     # 128 channels
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 6x6

            Fire(128, 32, 128, 128),   # 256 channels
            Fire(256, 32, 128, 128),   # 256 channels
            Fire(256, 48, 192, 192),   # 384 channels
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 3x3

            Fire(384, 48, 192, 192),   # 384 channels
            Fire(384, 64, 256, 256),   # 512 channels
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


# Alternative: More aggressive architecture for small images
class SqueezeNetCompact(nn.Module):
    """Optimized for 48x48 facial emotion recognition"""
    def __init__(self, num_classes=7):
        super(SqueezeNetCompact, self).__init__()
        
        self.features = nn.Sequential(
            # Initial block
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 48x48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24x24

            # Fire modules block 1
            Fire(64, 16, 64, 64),      # 128 channels
            Fire(128, 16, 64, 64),     # 128 channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12x12

            # Fire modules block 2
            Fire(128, 32, 128, 128),   # 256 channels
            Fire(256, 32, 128, 128),   # 256 channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6

            # Fire modules block 3
            Fire(256, 48, 192, 192),   # 384 channels
            Fire(384, 64, 256, 256),   # 512 channels
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)