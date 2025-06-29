import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=(2, 1), padding=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class DelayAwareUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 32, stride=(2, 1))   # 10240 -> 5120
        self.encoder2 = ConvBlock(32, 64, stride=(4, 1))  # 5120 -> 2560
        self.encoder3 = ConvBlock(64, 128, stride=(4, 1)) # 2560 -> 1280

        self.middle = ConvBlock(128, 256, stride=(4, 1))  # 1280 -> 320

        self.transformer = TransformerBlock(embed_dim=256, num_heads=8)

        self.up = lambda x: F.interpolate(x, scale_factor=(2, 1), mode='bilinear', align_corners=False)

        self.decoder3 = ConvBlock(256 + 128, 128, stride=1)
        self.decoder2 = ConvBlock(128 + 64, 64, stride=1)
        self.decoder1 = ConvBlock(64 + 32, 32, stride=4)

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 160),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        e1 = self.encoder1(x)                    # [B, 32, 5120, 4]
        e2 = self.encoder2(e1)                   # [B, 64, 2560, 4]
        e3 = self.encoder3(e2)                   # [B, 128, 1280, 4]
        m = self.middle(e3)                      # [B, 256, 320, 4]

        B, C, H, W = m.shape
        m_flat = m.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t_out = self.transformer(m_flat)
        m = t_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, 256, 320, 4]

        d3 = self.up(m)                          # [B, 256, 640, 4]
        d3 = self.decoder3(torch.cat([d3, F.interpolate(e3, size=(160, 4), mode='bilinear', align_corners=False)], dim=1))

        d2 = self.up(d3)                         # [B, 128, 1280, 4]
        d2 = self.decoder2(torch.cat([d2, F.interpolate(e2, size=(320, 4), mode='bilinear', align_corners=False)], dim=1))

        d1 = self.up(d2)                         # [B, 64, 2560, 4]
        d1 = self.decoder1(torch.cat([d1, F.interpolate(e1, size=(640, 4), mode='bilinear', align_corners=False)], dim=1))

        out = self.final(d1)                     # [B, 160]
        return out


if __name__ == '__main__':
    model = DelayAwareUNet()
    x = torch.randn(2, 3, 4, 10240)  # Reduce batch size if needed
    y = model(x)
    print(y.shape)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}\nTrainable parameters: {trainable:,}")

    count_parameters(model)