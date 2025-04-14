import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torch import Tensor


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


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.size_average = size_average

    def forward(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        return self.dice_loss(input, target, multiclass)

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        input = F.softmax(input, dim=1)
        target = F.one_hot(target, num_classes=input.size(1)
                           ).permute(0, 3, 1, 2).float()
        target = target.contiguous().view(input.size())
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(
            inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * \
            ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss
