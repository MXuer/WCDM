import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module): 
    def __init__(self, input_channels, num_channels, use_1x1conv=True, strides=1):
        super().__init__()
        # 使用1×16卷积核和"same"填充保持维度
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=(1, 16),
                               stride=strides)
        
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=(1, 16),
                               stride=strides)
        
        # 处理通道数变化的1×1卷积
        if use_1x1conv and input_channels != num_channels:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 主路径前向传播
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = F.relu(Y)
        
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        
        # 残差路径
        if self.conv3:
            X_res = self.conv3(X)
        else:
            X_res = X
        
        # 动态处理尺寸差异
        if Y.shape != X_res.shape:
            # 高度方向处理
            H_diff = Y.size(2) - X_res.size(2)
            if H_diff > 0:
                X_res = F.pad(X_res, (0, 0, 0, H_diff))  # 只在底部添加填充
            elif H_diff < 0:
                Y = F.pad(Y, (0, 0, 0, -H_diff))
            
            # 宽度方向处理
            W_diff = Y.size(3) - X_res.size(3)
            if W_diff > 0:
                X_res = F.pad(X_res, (0, W_diff, 0, 0))  # 只在右侧添加填充
            elif W_diff < 0:
                Y = F.pad(Y, (0, -W_diff, 0, 0))
        
        # 残差相加和激活
        Y += X_res
        return F.relu(Y)