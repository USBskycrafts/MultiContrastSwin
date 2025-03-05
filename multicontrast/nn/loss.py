import torch
import torch.nn as nn
from lpips import LPIPS


class CustomLPIPS(nn.Module):
    def __init__(self, input_channels=1, pretrained=True, train_adapter=True):
        super().__init__()
        # 定义适配层
        self.adapter = nn.Conv2d(
            input_channels, 3, kernel_size=1, stride=1, padding=0)

        # 控制适配层是否可训练
        if not train_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = False

        # 加载预训练的LPIPS（VGG）模型并冻结其所有参数
        self.lpips = LPIPS(net='vgg', pretrained=pretrained)
        for param in self.lpips.parameters():  # 冻结VGG参数
            param.requires_grad = False

    def forward(self, x, y):
        B, M, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * C * M, 1, H, W)
        y = y.permute(0, 4, 1, 2, 3).contiguous().view(B * C * M, 1, H, W)
        x = self.adapter(x)
        y = self.adapter(y)
        return self.lpips(x, y)
