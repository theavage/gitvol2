import torch.nn as nn

class Enc2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enc2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Enc3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enc3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Dec3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dec3, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class Dec2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dec2, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.enc1 = Enc2(in_channels=3, out_channels=64)
        self.enc2 = Enc2(in_channels=64, out_channels=128)
        self.enc3 = Enc3(in_channels=128, out_channels=256)
        self.enc4 = Enc3(in_channels=256, out_channels=512)
        self.enc5 = Enc3(in_channels=512, out_channels=1024)

        self.dec1 = Dec3(in_channels=1024, out_channels=512)
        self.dec2 = Dec3(in_channels=512, out_channels=256)
        self.dec3 = Dec3(in_channels=256, out_channels=128)
        self.dec4 = Dec2(in_channels=128, out_channels=64)
        self.dec5 = Dec2(in_channels=64, out_channels=1)
    
    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        
        return x