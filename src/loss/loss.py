import torch
import torch.nn as nn

import torch
import torch.nn as nn

# 定义自定义损失函数（结合MSE和二元交叉熵）
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        return bce_loss