import torch
import torch.nn as nn

class SignalCNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=160):
        super(SignalCNN, self).__init__()
        
        # 输入形状: (N, C_in, H, W) = (N, 3, 10240, 4)
        # 使用二维卷积网络处理信号数据
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_channels, 32, kernel_size=(7, 4), stride=(2, 1), padding=(3, 0)),
            # 输出: (N, 32, 5120, 1)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # 输出: (N, 32, 2560, 1)
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            # 输出: (N, 64, 1280, 1)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # 输出: (N, 64, 640, 1)
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            # 输出: (N, 128, 320, 1)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # 输出: (N, 128, 160, 1)
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            # 输出: (N, 256, 160, 1)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # 输出: (N, 256, 80, 1)
        )
        
        # 特征提取后的形状: (N, 256, 80, 1)
        # 使用自适应池化确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((40, 1))
        # 输出: (N, 256, 40, 1)
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平为 (N, 256*40*1) = (N, 10240)
            nn.Linear(256 * 40 * 1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # 用于二分类任务
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x 初始形状: (N, 3, 10240, 4)
        x = self.features(x)  # 输出: (N, 256, 80, 1)
        x = self.adaptive_pool(x)  # 输出: (N, 256, 40, 1)
        x = self.classifier(x)  # 输出: (N, 160)
        return x

# 示例用法（用于测试模型结构）:
if __name__ == '__main__':
    # 创建一个测试输入张量
    # batch_size = 10, channels = 3, height = 10240, width = 4
    test_input = torch.randn(10, 3, 10240, 4)
    
    # 实例化模型
    model = SignalCNN(input_channels=3, output_dim=160)
    
    # 将输入传递给模型
    output = model(test_input)
    
    # 打印输出形状进行验证
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")  # 预期: torch.Size([10, 160])
    
    # 检查输出值是否在0和1之间（由于Sigmoid激活函数）
    print(f"最小输出值: {output.min().item()}")
    print(f"最大输出值: {output.max().item()}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")