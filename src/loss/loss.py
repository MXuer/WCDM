import torch.nn as nn

# 定义自定义损失函数（结合MSE和二元交叉熵）
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # BCE权重
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        # 均方误差部分
        mse_loss = self.mse(predictions, targets)
        # 二元交叉熵部分
        bce_loss = self.bce(predictions, targets)
        # 加权组合
        return self.alpha * mse_loss + self.beta * bce_loss