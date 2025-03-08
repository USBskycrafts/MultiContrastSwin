import warnings
import torch.nn.functional as F
from typing import Callable, Optional, Sequence, Union
from typing import Callable, Dict, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["PSNR", "SSIM"]


class PSNR(Metric):
    # _state_dict_all_req_keys = ("_sum_of_batchwise_psnr", "_num_examples")
    _state_dict_all_req_keys = ("_mean", "_multipled_std", "_num_examples")

    def __init__(
        self,
        data_range: Union[int, float],
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(output_transform=output_transform,
                         device=device, skip_unrolling=skip_unrolling)
        self.data_range = data_range

    def _check_shape_dtype(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        # if y_pred.dtype != y.dtype:
        #     raise TypeError(
        #         f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
        #     )

        # if y_pred.shape != y.shape:
        #     raise ValueError(
        #         f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
        #     )

    @reinit__is_reduced
    def reset(self) -> None:
        self._mean = torch.tensor(
            0.0, dtype=torch.float64, device=self._device)
        self._multipled_std = torch.tensor(
            0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape_dtype(output)
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = y_pred.float()
        y = y.float()
        dim = tuple(range(1, y.ndim))
        mse_error = torch.pow(
            y_pred.double() - y.view_as(y_pred).double(), 2).mean(dim=dim)
        psnr = torch.sum(10.0 * torch.log10(self.data_range**2 / (mse_error + 1e-4))).to(
            device=self._device
        )
        _mean = self._mean
        self._num_examples += y.shape[0]
        self._mean += (psnr - _mean) / self._num_examples
        self._multipled_std += (psnr - _mean) * (psnr - self._mean)

    @sync_all_reduce("_mean", "_num_examples", "_multipled_std")
    def compute(self) -> Dict:
        if self._num_examples == 0:
            raise NotComputableError(
                "PSNR must have at least one example before it can be computed.")
        return {"mean": self._mean.item(), "std": torch.sqrt(self._multipled_std / self._num_examples).item()}


class SSIM(Metric):
    _state_dict_all_req_keys = (
        "_mean_of_ssim", "_std_of_ssim", "_num_examples", "_kernel")

    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if isinstance(kernel_size, int):
            self.kernel_size: Sequence[int] = [kernel_size, kernel_size]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma: Sequence[float] = [sigma, sigma]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError(
                "Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(
                f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(
                f"Expected sigma to have positive number. Got {sigma}.")

        super(SSIM, self).__init__(output_transform=output_transform,
                                   device=device, skip_unrolling=skip_unrolling)
        self.gaussian = gaussian
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel_2d = self._gaussian_or_uniform_kernel(
            kernel_size=self.kernel_size, sigma=self.sigma)
        self._kernel: Optional[torch.Tensor] = None

    @reinit__is_reduced
    def reset(self) -> None:
        self._mean_of_ssim = torch.tensor(
            0.0, dtype=torch.float64, device=self._device)
        self._std_of_ssim = torch.tensor(
            0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    def _uniform(self, kernel_size: int) -> torch.Tensor:
        kernel = torch.zeros(kernel_size)

        start_uniform_index = max(kernel_size // 2 - 2, 0)
        end_uniform_index = min(kernel_size // 2 + 3, kernel_size)

        min_, max_ = -2.5, 2.5
        kernel[start_uniform_index:end_uniform_index] = 1 / (max_ - min_)

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half,
                                steps=kernel_size, device=self._device)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size: Sequence[int], sigma: Sequence[float]) -> torch.Tensor:
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        # (kernel_size, 1) * (1, kernel_size)
        return torch.matmul(kernel_x.t(), kernel_y)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = y_pred.float()
        y = y.float()
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        # converts potential integer tensor to fp
        if not y.is_floating_point():
            y = y.float()
        if not y_pred.is_floating_point():
            y_pred = y_pred.float()

        nb_channel = y_pred.size(1)
        if self._kernel is None or self._kernel.shape[0] != nb_channel:
            self._kernel = self._kernel_2d.expand(nb_channel, 1, -1, -1)

        if y_pred.device != self._kernel.device:
            if self._kernel.device == torch.device("cpu"):
                self._kernel = self._kernel.to(device=y_pred.device)

            elif y_pred.device == torch.device("cpu"):
                warnings.warn(
                    "y_pred tensor is on cpu device but previous computation was on another device: "
                    f"{self._kernel.device}. To avoid having a performance hit, please ensure that all "
                    "y and y_pred tensors are on the same device.",
                )
                y_pred = y_pred.to(device=self._kernel.device)
                y = y.to(device=self._kernel.device)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w,
                       self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h,
                  self.pad_h], mode="reflect")

        if y_pred.dtype != self._kernel.dtype:
            self._kernel = self._kernel.to(dtype=y_pred.dtype)

        input_list = [y_pred, y, y_pred * y_pred, y * y, y_pred * y]
        outputs = F.conv2d(torch.cat(input_list),
                           self._kernel, groups=nb_channel)
        batch_size = y_pred.size(0)
        output_list = [
            outputs[x * batch_size: (x + 1) * batch_size] for x in range(len(input_list))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        ssim = torch.mean(ssim_idx, (1, 2, 3),
                          dtype=torch.float64).mean().to(device=self._device)
        _mean = self._mean_of_ssim
        self._num_examples += y.shape[0]
        self._mean_of_ssim += (ssim - _mean) / self._num_examples
        self._std_of_ssim += (ssim - _mean) * (ssim - self._mean_of_ssim)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_mean_of_ssim", "_std_of_ssim", "_num_examples")
    def compute(self) -> dict:
        if self._num_examples == 0:
            raise NotComputableError(
                "SSIM must have at least one example before it can be computed.")
        # return (self._sum_of_ssim / self._num_examples).item()
        return {"mean": self._mean_of_ssim.item(), "std": torch.sqrt(self._std_of_ssim / self._num_examples).item()}
