import torch
import torch.nn as nn
import torch.nn.functional as F

## Convolution Four
##***********************************************************************************************************
class ResBasic(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.four_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.four_conv(x)
        out = out + self.shortcut(x)
        return F.relu(out)



## Upscaling then double conv
##***********************************************************************************************************
class CatRes(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, bilinear=True):
        super().__init__()
        self.res = ResBasic(in_channels, out_channels, k_size)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.res(x)


## Output
##***********************************************************************************************************
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(OutConv, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=k_size, padding=int(k_size/2))
        self.conv_1 = nn.Conv2d(int(in_channels/2), out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_1(self.conv_3(x))
        

