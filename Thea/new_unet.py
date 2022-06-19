import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2d(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class new_UNet(nn.Module):
    def __init__(self):
        super(new_UNet, self).__init__()

        self.first_conv = DoubleConv2d(in_channels=3, out_channels=64)

        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        self.up1 = Up(in_channels=1024, out_channels=512, skip_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256, skip_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128, skip_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64, skip_channels=64)

        self.last_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):

        skip1 = self.first_conv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)

        x = self.down4(skip4)
        
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.last_conv(x)

        return x