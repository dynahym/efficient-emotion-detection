import torch
import torch.nn as nn


class ShiftConv(nn.Module):

    def __init__(self, in_channels, out_channels, shift_size=1):
        super().__init__()
        self.shift_size = shift_size
        self.in_channels = in_channels

        # Pointwise mixing conv (same as ShiftNet paper)
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # Divide channels into 4 groups
        g = 4
        assert C % g == 0, "Channels must be divisible by 4 for shifting"
        Cg = C // g

        # Split channels
        x1 = x[:, 0:Cg]               # shift up
        x2 = x[:, Cg:2*Cg]            # shift down
        x3 = x[:, 2*Cg:3*Cg]          # shift left
        x4 = x[:, 3*Cg:4*Cg]          # shift right

        # Apply spatial shifts
        x1 = torch.roll(x1, shifts=(-self.shift_size), dims=2)  # up
        x2 = torch.roll(x2, shifts=self.shift_size, dims=2)     # down
        x3 = torch.roll(x3, shifts=(-self.shift_size), dims=3)  # left
        x4 = torch.roll(x4, shifts=self.shift_size, dims=3)     # right

        # Concatenate back
        shifted = torch.cat([x1, x2, x3, x4], dim=1)

        return self.pw(shifted)


class ShiftNet(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            ShiftConv(32, 64),
            ShiftConv(64, 128),

            nn.MaxPool2d(2, 2),

            ShiftConv(128, 256),
            ShiftConv(256, 512),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
