import torch
from torch import nn
from torch.nn import functional as F
# from src.model.rescnn_net import Residual

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
        self.cnn_layers = nn.Sequential(
                Residual(3, 32, strides=2),
                # Residual(32, 32, strides=1),
                # Residual(32, 32, strides=1),
                Residual(32, 64, strides=2),
                # Residual(64, 64, strides=1),
                # Residual(64, 64, strides=1),
                Residual(64, 128, strides=2),
                # Residual(128, 128, strides=1),
                # Residual(128, 128, strides=1),
                Residual(128, 256, strides=2),
                # Residual(256, 256, strides=1),
                # Residual(256, 256, strides=1),
                Residual(256, 512, strides=2),
                Residual(512, 512, strides=1),
                Residual(512, 256, strides=1),
                # Residual(256, 256, strides=1),
                # Residual(256, 256, strides=1),
                Residual(256, 128, strides=2),
                # Residual(128, 128, strides=1),
                # Residual(128, 128, strides=1),
                Residual(128, 64, strides=2),
                # Residual(64, 64, strides=1),
                # Residual(64, 64, strides=1),
                Residual(64, 32, strides=2),
                # Residual(32, 32, strides=1),
                # Residual(32, 32, strides=1),
        )
        self.last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.last_layer(x)
        return x
    
    
if __name__=="__main__":
    # 创建残差块 (从3通道到32通道)
    input_channels = 3
    output_channels = 32
    block = Residual(input_channels, output_channels, use_1x1conv=True)
    
    # 创建测试输入数据 (批次=2, 通道=3, 高度=4, 宽度=10240)
    batch_size = 2
    height = 4
    width = 10240
    x = torch.randn(batch_size, input_channels, height, width)
    model = WCDMARESCNN()
    y = model(x)
    print(x.size())
    print(y.size())