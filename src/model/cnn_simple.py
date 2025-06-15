import torch
import torch.nn as nn

# 修改src/model/cnn.py中的模型结构
class SimpleSignalCNN(nn.Module):
    def __init__(self, input_channels=4, output_dim=160):
        super(SimpleSignalCNN, self).__init__()
        # 简化的2D卷积层
        self.conv2d_initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        
        # 简化的1D卷积层
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)  # 自适应池化到固定大小
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 降低dropout比例
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 处理输入形状
        x = self.conv2d_initial(x)
        # 调整维度以适应1D卷积
        x = x.view(x.size(0), x.size(1), -1)
        x = self.conv1d_layers(x)
        x = self.fc_layers(x)
        return x