from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn

from multicontrast.nn.model import MultiContrastSwinTransformer


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        self.predict(*args, **kwargs)

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
        self.loss_fn = nn.L1Loss()

    def loss(self, x, selected_contrasts, y=None, lambdas: List[int] = [5, 20]):
        contrasts = [i for i in range(self.model.num_contrats)]
        generated_contrasts = [
            c for c in contrasts if c not in selected_contrasts]
        if y is None:
            y = x[:, generated_contrasts, ...]
            x = x[:, selected_contrasts, ...]

        pred = self.model(x, [selected_contrasts, generated_contrasts])
        recon = self.model(x, [selected_contrasts, selected_contrasts])
        return self.loss_fn(pred, y) * lambdas[1] + self.loss_fn(recon, x) * lambdas[0]

    def predict(self, x, selected_contrasts: List[int]):
        contrasts = [i for i in range(self.model.num_contrats)]
        generated_contrasts = [
            c for c in contrasts if c not in selected_contrasts]
        return self.model(x, [selected_contrasts, generated_contrasts])
