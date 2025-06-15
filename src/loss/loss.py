import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9):  # alpha 控制 MSE, beta 控制 BCE
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, return_parts=False):
        # BCE: 直接用 logits
        bce_loss = self.bce(predictions, targets)

        # MSE: 对 logits 先过 sigmoid（概率），再计算 MSE
        if self.alpha > 0:
            probs = torch.sigmoid(predictions)
            mse_loss = self.mse(probs, targets)
        else:
            mse_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = self.alpha * mse_loss + self.beta * bce_loss

        if return_parts:
            return total_loss, mse_loss.detach(), bce_loss.detach()
        else:
            return total_loss