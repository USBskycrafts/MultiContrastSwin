from ignite.metrics import PSNR, SSIM
import torch


class StablePSNR(PSNR):
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = y_pred.double()
        y = y.double()
        if torch.sum(y_pred) < 1e-2 and torch.sum(y) < 1e-2:
            return
        super().update([y_pred, y])


class StableSSIM(SSIM):
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = y_pred.double()
        y = y.double()
        if torch.sum(y_pred - y) < 1e-2 and torch.sum(y) < 1e-2:
            return
        super().update([y_pred, y])
