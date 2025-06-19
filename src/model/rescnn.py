import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module): 
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=True, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class WCDMARESCNN(nn.Module):
    def __init__(self, input_height=4, input_width=10240, input_channels=3, output_size=160):
        super(WCDMARESCNN, self).__init__()
        self.layer1 = Residual(3, 32, strides=2)
        self.layer2 = Residual(32, 64, strides=2)
        self.layer3 = Residual(64, 128, strides=2)
        self.layer4 = Residual(128, 256, strides=2)
        self.layer5 = Residual(256, 128, strides=2)
        self.layer6 = Residual(128, 64, strides=2)
        self.layer7 = Residual(64, 32, strides=2)
        self.last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.last_layer(x)
        return x