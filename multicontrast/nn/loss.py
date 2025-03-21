import torch
import torch.nn as nn
from lpips import LPIPS


class CustomLPIPS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练的LPIPS（VGG）模型并冻结其所有参数
        self.lpips = LPIPS(pretrained=pretrained)
        for param in self.lpips.parameters():  # 冻结VGG参数
            param.requires_grad = False

    def forward(self, x, y):
        B, M, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W).expand(-1, 3, -1, -1)
        y = y.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W).expand(-1, 3, -1, -1)
        return self.lpips(x, y)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, x, y):
        B, M, H, W, C = x.shape
        assert x.shape == y.shape, "Input shapes must match"
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W)
        y = y.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W)
        return self.loss_fn(x, y)
