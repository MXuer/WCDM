import torch
import torch.nn as nn

class WCDMAFingerCNN4(nn.Module):
    def __init__(self, input_channels=3, feature_height=4, seq_len=160):
        super(WCDMAFingerCNN4, self).__init__()
        self.seq_len = seq_len
        
        # 优化后的卷积层结构（明确指定步长）
        self.conv_layers = nn.Sequential(
            # 第一层: 特征维度保持，序列维度保持
            nn.Conv2d(input_channels, 64, kernel_size=(2, 1), stride=(2,1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 第二层: 保持所有维度
            nn.Conv2d(64, 128, kernel_size=(2, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层: 保持所有维度
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第四层: 使用稍大的核增加感受野
            nn.Conv2d(256, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第五层: 保持所有维度
            nn.Conv2d(128, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.1),
            
            # 第六层: 压缩特征维度到1，保持序列维度
            nn.Conv2d(64, 32, kernel_size=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # 动态计算全连接层输入尺寸
        self._init_fc_layers()
        
    def _init_fc_layers(self):
        # 创建测试输入
        test_input = torch.zeros(2, 3, 4, self.seq_len)
        
        # 通过卷积层获取输出尺寸
        conv_output = self.conv_layers(test_input)
        flattened_size = conv_output.numel() // test_input.shape[0]  # 计算每个样本的特征数
        
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 160),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入x形状: [batch, 3, 4, 160]
        x = self.conv_layers(x)  # 卷积特征提取
        x = self.fc_layers(x)    # 全连接处理
        return x

# 测试验证
if __name__ == "__main__":
    model = WCDMAFingerCNN4()
    x = torch.randn(16, 3, 4, 160)
    y = model(x)
    print("输入尺寸:", x.shape)    # [16, 3, 4, 160]
    print("输出尺寸:", y.shape)    # [16, 160]