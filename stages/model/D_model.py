"""
把三个路径输出的 (B, 170, 2) 的输出拼接起来，得到 (B, 3, 170, 2) 经过一个很小的Unet，和全连接输出，得到 (B，160) 的输出
这个模型的Loss是BinaryCrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stages.model.delay_unet import TransformerBlock
from stages.model.delay_unet import ConvBlock
from stages.model.delay_unet import count_parameters



class DUNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(1, 32, stride=(4, 1))   # 10240 -> 5120
        self.encoder2 = ConvBlock(32, 64, stride=(4, 1))  # 5120 -> 2560

        self.middle = ConvBlock(64, 64, stride=(2, 1))  # 1280 -> 320

        self.transformer = TransformerBlock(embed_dim=64, num_heads=4)

        self.up = lambda x: F.interpolate(x, scale_factor=(2, 1), mode='bilinear', align_corners=False)

        self.decoder2 = ConvBlock(64 + 64, 64, stride=2)
        self.decoder1 = ConvBlock(64 + 32, 32, stride=2)
        self.final = ConvBlock(32, 2, stride=(1, 1))  # 1280 -> 320


    def forward(self, x):
        e1 = self.encoder1(x)                    # [B, 32, 5120, 4]
        e2 = self.encoder2(e1)                   # [B, 64, 2560, 4]
        m = self.middle(e2)                      # [B, 256, 320, 4]

        B, C, H, W = m.shape
        m_flat = m.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t_out = self.transformer(m_flat)
        m = t_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, 256, 320, 4]
        d2 = self.up(m)                         # [B, 128, 1280, 4]
        d2 = self.decoder2(torch.cat([d2, F.interpolate(e2, size=(320, 4), mode='bilinear', align_corners=False)], dim=1))

        d1 = self.up(d2)                         # [B, 64, 2560, 4]
        d1 = self.decoder1(torch.cat([d1, F.interpolate(e1, size=(320, 2), mode='bilinear', align_corners=False)], dim=1))
        out = self.final(d1).squeeze(3)                     # [B, 160]
        return out        
    
    
if __name__=="__main__":
    x = torch.randn(1, 1, 5120, 4)  # Reduce batch size if needed
    model = DUNET()
    print(x.shape)
    y = model(x)
    print(y.shape)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}\nTrainable parameters: {trainable:,}")

    count_parameters(model)
    
    criterion = nn.MSELoss()  # 二元交叉熵损失，适用于0/1输出
    output = torch.randn(1, 2, 160)
    criterion = nn.MSELoss()
    print(criterion(y, output).item())