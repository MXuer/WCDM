import torch
import torch.nn as nn

# 定义CNN模型架构
# 修改后的CNN模型架构（添加Dropout和正则化）
class WCDMACNNC4(nn.Module):
    def __init__(self, input_height=3, input_width=10240, input_channels=4, output_size=160):
        super(WCDMACNNC4, self).__init__()
        
        # 修改后的卷积层序列（添加Dropout）
        self.conv_layers = nn.Sequential(
            # 第一卷积层
            nn.Conv2d(input_channels, 64, kernel_size=(2, 64), stride=(2, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二卷积层
            nn.Conv2d(64, 128, kernel_size=(1, 16), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三卷积层
            nn.Conv2d(128, 256, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加30%的Dropout
            
            # 第四卷积层
            nn.Conv2d(256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加20%的Dropout
            
            # 第五卷积层
            nn.Conv2d(128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.LeakyReLU(0.1),
            
            # 第六卷积层
            nn.Conv2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # 展平后计算特征大小
        self._init_dense(input_height, input_width, input_channels, output_size)
        
    def _init_dense(self, input_height, input_width, input_channels, output_size):
        # 创建两个样本作为测试输入
        test_input = torch.zeros(2, input_channels, input_height, input_width)
        
        # 设置模型为评估模式进行测试
        original_mode = self.conv_layers.training
        self.conv_layers.eval()
        with torch.no_grad():
            conv_output = self.conv_layers(test_input)
        self.conv_layers.train(original_mode)
        
        flattened_size = conv_output.view(2, -1).shape[1]
        
        # 修改后的全连接层序列（添加更多Dropout）
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1000),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.Dropout(0.5),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x
