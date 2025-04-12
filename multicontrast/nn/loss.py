import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, pred, target, alpha=1.0, beta=1.0):
        B, M, H, W, C = pred.shape
        assert pred.shape == target.shape, "Input shapes must match"
        pred = pred.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W)
        target = target.permute(0, 4, 1, 2, 3).contiguous().view(
            B * C * M, 1, H, W)
       # 基础误差（如绝对值误差）
        error_map = torch.abs(pred - target)

        # 计算局部信息量（以梯度为例）
        gradient = self.compute_gradient(target)  # 或用 pred 计算
        info_weight = gradient / (gradient.max() + 1e-10)  # 归一化到 [0,1]

        # 结合误差与信息量的权重
        combined_weight = alpha + beta * info_weight
        combined_weight = combined_weight.detach()  # 阻止梯度传播到权重

        # 加权损失
        loss = torch.mean(combined_weight * error_map)
        return loss

    def compute_gradient(self, image):
        # Sobel 算子计算梯度
        sobel_x = torch.tensor(
            [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=image.dtype, device=image.device)
        sobel_y = torch.tensor(
            [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=image.dtype, device=image.device)
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude
