from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn

from multicontrast.nn.model import MultiContrastSwinTransformer, MultiContrastDiscriminator
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
        self.perceptual_loss = CustomLPIPS()
        self.l1_loss = L1Loss()

    def loss(self, x, selected_contrasts, generated_contrasts, y, sample_times=1):
        pred = self.model(
            x, [selected_contrasts, generated_contrasts], sample_times=sample_times)
        # recon = self.model(
        #     x, [selected_contrasts, selected_contrasts], sample_times=sample_times)
        # * lambdas[1] + self.loss_fn(recon, x) * lambdas[0]
        return self.l1_loss(pred, y) * 0.5 + self.perceptual_loss(pred, y).mean() * 0.5

    def predict(self, x, selected_contrasts: List[int], generated_contrasts, sample_times=1):
        return self.model(x, [selected_contrasts, generated_contrasts], sample_times=sample_times)


class MultiContrastDiscrimination(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MultiContrastDiscriminator(*args, **kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def loss(self, x, generated_contrasts, label):
        pred = self.model(x, generated_contrasts)
        if label.item() == 1:
            return self.loss_fn(pred, torch.ones_like(pred))
        elif label.item() == 0:
            return self.loss_fn(pred, torch.zeros_like(pred))
        else:
            raise ValueError("Label must be 0 or 1")

    def predict(self, x, generated_contrasts):
        return self.model(x, generated_contrasts)
