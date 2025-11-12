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
    def __init__(self, num_classes=7, in_channels=1):

        super(SqueezeNet, self).__init__()
        
        # Initial convolution layer
        # Input: 1x48x48 -> Output: 96x23x23 (with stride=2)
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire modules
        # After maxpool1: 96x12x12
        self.fire2 = Fire(96, 16, 64, 64)      # Output: 128x12x12
        self.fire3 = Fire(128, 16, 64, 64)     # Output: 128x12x12
        self.fire4 = Fire(128, 32, 128, 128)   # Output: 256x12x12
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # After maxpool4: 256x6x6
        self.fire5 = Fire(256, 32, 128, 128)   # Output: 256x6x6
        self.fire6 = Fire(256, 48, 192, 192)   # Output: 384x6x6
        self.fire7 = Fire(384, 48, 192, 192)   # Output: 384x6x6
        self.fire8 = Fire(384, 64, 256, 256)   # Output: 512x6x6
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # After maxpool8: 512x3x3
        self.fire9 = Fire(512, 64, 256, 256)   # Output: 512x3x3
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Final convolution layer
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate methods."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution block
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        
        # Fire modules with pooling
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        
        x = self.fire9(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Final classification
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        return x