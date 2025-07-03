import torch
import torch.nn as nn

class WCDMACNNDISTRACT(nn.Module):
    def __init__(self, input_height=6, input_width=64, input_channels=1, output_size=1):
        super(WCDMACNNDISTRACT, self).__init__()
        
        # 更新卷积层参数以适应64像素的输入宽度
        self.conv_layers = nn.Sequential(
            # 第一层：kernel_size和padding调整
            nn.Conv2d(input_channels, 64, kernel_size=(2, 16), stride=(2, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层：减小卷积核尺寸
            nn.Conv2d(64, 128, kernel_size=(2, 8), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层：进一步减小卷积核
            nn.Conv2d(128, 256, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1)),  # stride改为1防止过快压缩
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第四层：保持较小卷积核
            nn.Conv2d(256, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第五层：更小的卷积核
            nn.Conv2d(128, 64, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0)),  # 移除padding
            nn.LeakyReLU(0.1),
            
            # 第六层：最后处理层
            nn.Conv2d(64, 32, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # 动态计算全连接层输入尺寸
        self._init_dense(input_height, input_width, input_channels, output_size)
        
    def _init_dense(self, input_height, input_width, input_channels, output_size):
        # 创建测试输入确定特征向量尺寸
        test_input = torch.zeros(2, input_channels, input_height, input_width)
        
        # 计算卷积输出尺寸
        original_mode = self.conv_layers.training
        self.conv_layers.eval()
        with torch.no_grad():
            conv_output = self.conv_layers(test_input)
        self.conv_layers.train(original_mode)
        
        flattened_size = conv_output.view(2, -1).shape[1]
        
        # 更新全连接层以适应1维输出
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),  # 减小层大小
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.4),  # 调整dropout率
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_size),  # 直接输出最终结果
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x
