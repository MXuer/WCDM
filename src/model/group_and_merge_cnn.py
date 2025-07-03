import torch
import torch.nn as nn

class MultiGroupCNNModel(nn.Module):
    """
    完整的多分组CNN模型，封装所有功能：
    1. 三组独立的CNN处理，每组包含两个不同结构的CNN
    2. 自动处理不同批次的输入
    3. 手动合并组内输出
    4. 最终融合模型输出单路信号

    待解决问题：如果输入的batch是不同的，输出的loss的计算的batch怎么控制？
    
    输入格式:
        (group1_input1, group1_input2,
         group2_input1, group2_input2,
         group3_input1, group3_input2)
        
        每个输入形状: [batch_size, sequence_length]
    """
    
    def __init__(self, 
                 group1_cnn1, group1_cnn2,
                 group2_cnn1, group2_cnn2,
                 group3_cnn1, group3_cnn2,
                 fusion_model):
        """
        初始化多分组CNN模型
        
        参数:
            group1_cnn1: 第一组的第一个CNN模型
            group1_cnn2: 第一组的第二个CNN模型
            group2_cnn1: 第二组的第一个CNN模型
            group2_cnn2: 第二组的第二个CNN模型
            group3_cnn1: 第三组的第一个CNN模型
            group3_cnn2: 第三组的第二个CNN模型
            fusion_model: 最终的融合模型
        """
        super().__init__()
        
        # 第一组CNN模型
        self.group1_cnn1 = group1_cnn1
        self.group1_cnn2 = group1_cnn2
        
        # 第二组CNN模型
        self.group2_cnn1 = group2_cnn1
        self.group2_cnn2 = group2_cnn2
        
        # 第三组CNN模型
        self.group3_cnn1 = group3_cnn1
        self.group3_cnn2 = group3_cnn2
        
        # 融合模型
        self.fusion = fusion_model
    
    def forward(self, *inputs):
        """
        前向传播，处理6个输入信号
        
        参数:
            *inputs: 6个输入信号，顺序为:
                group1_input1, group1_input2,
                group2_input1, group2_input2,
                group3_input1, group3_input2
                
        返回:
            输出信号 [batch, 1]
        """
        
        # 处理第一组
        g1_out1 = self.group1_cnn1(inputs[0])
        g1_out2 = self.group1_cnn2(inputs[1])
        
        # 处理第二组
        g2_out1 = self.group2_cnn1(inputs[2])
        g2_out2 = self.group2_cnn2(inputs[3])
        
        # 处理第三组
        g3_out1 = self.group3_cnn1(inputs[4])
        g3_out2 = self.group3_cnn2(inputs[5])
        
        
        # 拼合所有输出特征
        combined = torch.cat([
            g1_out1, g1_out2,
            g2_out1, g2_out2,
            g3_out1, g3_out2
        ], dim=1)
        
        
        # 通过融合模型
        return self.fusion(combined)
    
    @classmethod
    def create_example_model(cls, output_dim=64, fusion_dim=256):
        """
        创建示例模型（用于演示）
        
        参数:
            output_dim: 每个CNN模型的输出维度
            fusion_dim: 融合模型的输入维度
            
        返回:
            预配置的MultiGroupCNNModel实例
        """
        # 定义第一组的CNN模型
        class Group1CNN1(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Conv1d(1, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(64, output_dim)
                )
            
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                return self.model(x)
        
        class Group1CNN2(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Conv1d(1, 64, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, 5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                    nn.Flatten(),
                    nn.Linear(128, output_dim)
                )
            
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                return self.model(x)
        
        # 第二组复用第一组的模型结构
        # (但使用独立实例，参数不共享)
        group1_cnn1 = Group1CNN1()
        group1_cnn2 = Group1CNN2()
        
        # 第三组复用第一组的模型结构
        group2_cnn1 = Group1CNN1()
        group2_cnn2 = Group1CNN2()
        group3_cnn1 = Group1CNN1()
        group3_cnn2 = Group1CNN2()
        
        # 融合模型
        fusion_model = nn.Sequential(
            nn.Linear(6 * output_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        return cls(
            group1_cnn1, group1_cnn2,
            group2_cnn1, group2_cnn2,
            group3_cnn1, group3_cnn2,
            fusion_model
        )


# ======================
# 使用示例
# ======================

if __name__ == "__main__":
    # 创建模型实例
    model = MultiGroupCNNModel.create_example_model()
    
    print("模型结构:")
    print(model)
    
    # 准备不同批次的测试输入
    seq_len = 100
    group1_input1 = torch.randn(10, seq_len)  # batch=10
    group1_input2 = torch.randn(8, seq_len)   # batch=8
    group2_input1 = torch.randn(12, seq_len)  # batch=12
    group2_input2 = torch.randn(9, seq_len)   # batch=9
    group3_input1 = torch.randn(11, seq_len)  # batch=11
    group3_input2 = torch.randn(10, seq_len)  # batch=10
    
    # 调用模型 (只需要按顺序传递6个参数)
    output = model(
        group1_input1, group1_input2,
        group2_input1, group2_input2,
        group3_input1, group3_input2
    )
    
    print("\n输出形状:", output.shape)  # 应为 [min_batch, 1] = [8, 1]
    
    # 测试不同序列长度
    print("\n测试不同序列长度:")
    diff_len_inputs = [
        torch.randn(10, 80),   # 80长度
        torch.randn(8, 100),    # 100长度
        torch.randn(12, 120),   # 120长度
        torch.randn(9, 90),     # 90长度
        torch.randn(11, 110),   # 110长度
        torch.randn(10, 95)     # 95长度
    ]
    
    diff_len_output = model(*diff_len_inputs)
    print("不同长度输入输出形状:", diff_len_output.shape)  # [8, 1]
    
    # 保存和加载模型
    print("\n保存和加载模型测试:")
    torch.save(model.state_dict(), "multigroup_cnn_model.pth")
    
    # 创建新模型实例
    new_model = MultiGroupCNNModel.create_example_model()
    new_model.load_state_dict(torch.load("multigroup_cnn_model.pth"))
    
    # 测试加载的模型
    reload_output = new_model(
        group1_input1, group1_input2,
        group2_input1, group2_input2,
        group3_input1, group3_input2
    )
    print("重新加载的输出形状:", reload_output.shape)