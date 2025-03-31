from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn

from multicontrast.nn.model import MultiContrastSwinTransformer, MultiScaleDiscriminator
from multicontrast.nn.loss import CustomLPIPS, L1Loss


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

    def loss(self, x, selected_contrasts, generated_contrasts, y, sample_times=1):
        pred = self.model(
            x, [selected_contrasts, generated_contrasts], sample_times=sample_times)
        # recon = self.model(
        #     x, [selected_contrasts, selected_contrasts], sample_times=sample_times)
        # * lambdas[1] + self.loss_fn(recon, x) * lambdas[0]
        return self.l1_loss(pred, y), pred

    def predict(self, x, selected_contrasts: List[int], generated_contrasts, sample_times=1):
        return self.model(x, [selected_contrasts, generated_contrasts], sample_times=sample_times)


class MultiContrastDiscrimination(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MultiScaleDiscriminator(
            input_nc=1, ndf=64, n_layers=3, num_scales=5)
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
        loss = 0
        for pred in preds:
            if label.item() == 1:
                loss += self.loss_fn(pred, torch.ones_like(pred))
            elif label.item() == 0:
                loss += self.loss_fn(pred, torch.zeros_like(pred))
            else:
                raise ValueError("Label must be 0 or 1")
        return loss / len(preds)

    def predict(self, x, generated_contrasts):
        x = self._reshape_input(x)  # (B*M, C, H, W)
        return self.model(x)
