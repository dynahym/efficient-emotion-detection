import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------- MobileNetV1 ----------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Conv / s2
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv dw / s1
            DepthwiseSeparableConv(32, 64, 1),

            # Conv dw / s2
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),

            # Conv dw / s2
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),

            # Conv dw / s2
            DepthwiseSeparableConv(256, 512, 2),

            # 5× (Conv dw / s1)
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),

            # Conv dw / s2
            DepthwiseSeparableConv(512, 1024, 2),

            # Conv dw / s1
            DepthwiseSeparableConv(1024, 1024, 1),

            # Avg Pool / s1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)



# ---------- MobileNetV2 ----------

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expansion phase (only if expand_ratio != 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Projection phase (linear bottleneck - no activation)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, in_channels=3, width_mult=1.0):
        super().__init__()
        
        # Building first layer
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        # First conv layer
        self.features = [
            nn.Sequential(
                nn.Conv2d(in_channels, input_channel, kernel_size=3, 
                         stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True)
            )
        ]
        
        # Inverted residual blocks configuration
        # t: expansion factor, c: output channels, n: repeat times, s: stride
        inverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],   # Bottleneck 1
            [6, 24, 2, 2],   # Bottleneck 2-3
            [6, 32, 3, 2],   # Bottleneck 4-6
            [6, 64, 4, 2],   # Bottleneck 7-10
            [6, 96, 3, 1],   # Bottleneck 11-13
            [6, 160, 3, 2],  # Bottleneck 14-16
            [6, 320, 1, 1],  # Bottleneck 17
        ]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        # Building last several layers
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True)
            )
        )
        
        # Combine all feature layers
        self.features = nn.Sequential(*self.features)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# ---------- MobileNetV3 ----------

class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class h_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeezed = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, in_channels, 1),
            h_sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, expansion,
                 use_se, activation):
        super().__init__()

        hidden_dim = in_channels * expansion

        # Activation type
        act = nn.ReLU(inplace=True) if activation == "RE" else h_swish()

        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Expansion
        if expansion != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act
            ]

        # Depthwise
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride,
                      padding=kernel//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act
        ]

        # Squeeze-and-Excitation
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # Projection
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=7, in_channels=1, mode="small"):
        super().__init__()

        assert mode in ["small", "large"]

        # Configurations based on the official MobileNetV3 paper
        if mode == "small":
            cfgs = [
                # k, exp,  out, se, act, s
                [3,  16,  16,  True,  "RE", 2],
                [3,  72,  24,  False, "RE", 2],
                [3,  88,  24,  False, "RE", 1],

                [5,  96,  40,  True,  "HS", 2],
                [5, 240,  40,  True,  "HS", 1],
                [5, 240,  40,  True,  "HS", 1],

                [5, 120,  48,  True,  "HS", 1],
                [5, 144,  48,  True,  "HS", 1],

                [5, 288,  96,  True,  "HS", 2],
                [5, 576,  96,  True,  "HS", 1],
                [5, 576,  96,  True,  "HS", 1],
            ]
            last_channel = 576

        else:  # large
            cfgs = [
                # k, exp,  out, se, act, s
                [3,  16,  16,  False, "RE", 1],
                [3,  64,  24,  False, "RE", 2],
                [3,  72,  24,  False, "RE", 1],

                [5,  72,  40,  True,  "RE", 2],
                [5, 120,  40,  True,  "RE", 1],
                [5, 120,  40,  True,  "RE", 1],

                [3, 240,  80,  False, "HS", 2],
                [3, 200,  80,  False, "HS", 1],
                [3, 184,  80,  False, "HS", 1],
                [3, 184,  80,  False, "HS", 1],

                [3, 480, 112,  True,  "HS", 1],
                [3, 672, 112,  True,  "HS", 1],

                [5, 672, 160,  True,  "HS", 2],
                [5, 960, 160,  True,  "HS", 1],
                [5, 960, 160,  True,  "HS", 1],
            ]
            last_channel = 960

        # First convolution
        input_channel = 16
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            h_swish()
        )

        # Build blocks
        layers = []
        for k, exp, out_c, se, act, s in cfgs:
            layers.append(
                MobileNetV3Block(input_channel, out_c, k, s, exp, se, act)
            )
            input_channel = out_c

        self.blocks = nn.Sequential(*layers)

        # Final layers
        self.final_expand = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            h_swish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_expand(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)