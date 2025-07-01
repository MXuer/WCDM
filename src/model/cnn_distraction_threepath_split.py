import torch
import torch.nn as nn

class WCDMACNNDISTRACT(nn.Module):
    def __init__(self, input_height=4, input_width=64, input_channels=3, output_size=1):
        super(WCDMACNNDISTRACT, self).__init__()
        
        # 更新卷积层以适应4像素的高度和3通道输入
        self.conv_layers = nn.Sequential(
            # 第一层：调整kernel_size适应较小高度，修改输入通道为3
            nn.Conv2d(input_channels, 64, kernel_size=(2, 16), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层：减小卷积核高度避免尺寸过小
            nn.Conv2d(64, 128, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),  # 高度方向使用1x卷积
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层：保持高度不变
            nn.Conv2d(128, 256, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第四层：保持高度不变
            nn.Conv2d(256, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第五层：减少输出通道
            nn.Conv2d(128, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.LeakyReLU(0.1),
            
            # 第六层：调整为合适的输出尺寸
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
        
        # 更新全连接层
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x