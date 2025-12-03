import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, -1, height, width)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, inp, outp, stride):
        super().__init__()
        self.stride = stride

        branch_features = outp // 2
        assert self.stride in [1, 2]

        if stride == 1:
            # x is split into 2 branches
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),

                nn.Conv2d(branch_features, branch_features, 3, 1, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),

                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            # stride = 2 → no split
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 2, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),

                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),

                nn.Conv2d(branch_features, branch_features, 3, 2, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),

                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=7, in_channels=1, model_size="1.0x"):
        super().__init__()

        sizes = {
            "0.5x": [24, 48, 96, 192, 1024],
            "1.0x": [24, 116, 232, 464, 1024],
            "1.5x": [24, 176, 352, 704, 1024],
            "2.0x": [24, 244, 488, 976, 2048],
        }

        if model_size not in sizes:
            raise ValueError("Choose from 0.5x, 1.0x, 1.5x, 2.0x")

        out_channels = sizes[model_size]

        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Stage repeats
        repeats = [4, 8, 4]
        input_channels = out_channels[0]
        blocks = []
        for stage, repeat in enumerate(repeats):
            output_channels = out_channels[stage + 1]
            for i in range(repeat):
                stride = 2 if i == 0 else 1
                blocks.append(ShuffleBlock(input_channels, output_channels, stride))
                input_channels = output_channels
        self.features = nn.Sequential(*blocks)

        # Final layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels[-1]),
            nn.ReLU(inplace=True)
        )

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels[-1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
