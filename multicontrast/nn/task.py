import sys
from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
from ignite.utils import setup_logger

from multicontrast.nn.loss import CustomLPIPS, L1Loss
from multicontrast.nn.model import (MultiContrastSwinTransformer,
                                    MultiScaleDiscriminator)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.loss(*args, **kwargs)
        else:
            return self.predict(*args, **kwargs)

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class MultiModalityGeneration(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MultiContrastSwinTransformer(*args, **kwargs)
        self.l1_loss = L1Loss()
        self.percep_loss = CustomLPIPS()
        self.l1_weight = 1  # 初始权重
        self.decay_rate = 2e-4  # 衰减率

        self.register_full_backward_hook(lambda *_: self.update_l1_weight())
        self.logger = setup_logger(__name__, stream=sys.stdout)

    def update_l1_weight(self):
        """在每个epoch结束时调用，衰减L1权重"""
        # self.l1_weight = max(1.0, self.l1_weight - self.decay_rate)
        self.logger.info(f'''
                               L1 Weight: {self.l1_weight}
                               L1 Loss {self.l1_log} 
                               Perceptual Loss {self.per_log} 
                               Quantization  Loss {self.q_log} 
                         ''')

    def loss(self, x, selected_contrasts, generated_contrasts, y, sample_times=1):
        q_loss, pred = self.model(
            x, [selected_contrasts, generated_contrasts], sample_times=sample_times)
        l1_loss = self.l1_loss(pred, y)
        percep_loss = self.percep_loss(pred, y).mean()
        self.l1_log = l1_loss.item()
        self.per_log = percep_loss.item()
        self.q_log = q_loss.item()
        return l1_loss * self.l1_weight + percep_loss + q_loss, pred

    def predict(self, x, selected_contrasts: List[int], generated_contrasts, sample_times=1):
        return self.model(x, [selected_contrasts, generated_contrasts], sample_times=sample_times)[1]


class MultiContrastDiscrimination(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MultiScaleDiscriminator(
            input_nc=1, ndf=64, n_layers=3, num_scales=3)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _reshape_input(self, x):
        # x: (B, M, H, W, C)
        B, M, H, W, C = x.shape
        assert C == 1, "Input channels must be 1"
        return x.permute(0, 1, 4, 2, 3).contiguous()\
            .view(-1, C, H, W)

    def loss(self, x, generated_contrasts, label):
        x = self._reshape_input(x)  # (B*M, C, H, W)
        preds = self.model(x)
        # label: tensor(1.0) or tensor(0.0)
        # Flatten predictions for loss calculation
        preds = [pred.flatten() for pred in preds]
        # Concatenate predictions for loss calculation
        preds = torch.cat(preds, dim=0)
        # Expand label to match predictions shape
        label = label.expand_as(preds)
        loss = self.loss_fn(preds, label)  # Calculate loss
        return loss

    def predict(self, x, generated_contrasts):
        x = self._reshape_input(x)  # (B*M, C, H, W)
        return self.model(x)
