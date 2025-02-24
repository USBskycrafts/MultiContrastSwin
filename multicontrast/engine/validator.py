import os
from abc import ABCMeta, abstractmethod

import ignite.distributed as distributed
import torch
from ignite.distributed import auto_dataloader, auto_model
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from skimage.io import imsave
from torch.cuda.amp.autocast_mode import autocast

from multicontrast.utils.metrics import StablePSNR as PSNR
from multicontrast.utils.metrics import StableSSIM as SSIM


class BaseValidator(metaclass=ABCMeta):
    def __init__(self):
        # TODO: load model weights
        self.engine = Engine(lambda engine, batch: self._validate_step(batch))

    def validate(self, data_loader):
        data_loader = auto_dataloader(data_loader)
        psnr = PSNR(data_range=1, device=distributed.device(),
                    output_transform=lambda y: (y[0].squeeze(-1), y[1].squeeze(-1)))
        ssim = SSIM(data_range=1, device=distributed.device(),
                    output_transform=lambda y: (y[0].squeeze(-1), y[1].squeeze(-1)))
        psnr.attach(self.engine, name="psnr")
        ssim.attach(self.engine, name="ssim")
        self.register_events(Events.COMPLETED, lambda *_: print(
            f"PSNR: {self.engine.state.metrics['psnr']:.4f}, SSIM: {self.engine.state.metrics['ssim']:.4f}"
        ))
        self.engine.run(data_loader, 1)
        return self.engine.state.metrics

    @abstractmethod
    def _validate_step(self, batch):
        raise NotImplementedError("Subclasses must implement this method")

    def register_events(self, event_name, handler):
        self.engine.add_event_handler(event_name, handler)

    @abstractmethod
    def load_environment(self, checkpoint):
        raise NotImplementedError("Subclasses must implement this method")


class SupervisedValidator(BaseValidator):
    def __init__(self, model, save_images=False, output_dir=None):
        super().__init__()
        self.model = auto_model(model)
        self.save_images = save_images
        self.output_dir = output_dir

    def _validate_step(self, batch):
        self.model.eval()
        x = batch['x'].to(distributed.device())
        y = batch['y'].to(distributed.device())
        selected_contrasts = batch['selected_contrasts']
        generated_contrats = batch['generated_contrasts']
        with torch.no_grad():
            with autocast():
                pred = self.model(x, selected_contrasts, generated_contrats)

        if self.save_images and self.output_dir is not None:
            # Save images for visualization
            # This is a placeholder for actual image saving logic
            filenames = batch['filenames']
            for i, (filename, images) in enumerate(zip(filenames, pred)):
                for j, img in enumerate(images):
                    sample = filename + f'_pred_{j}.png'
                    sample = os.path.join(self.output_dir, sample)
                    imsave(sample, img)

        return pred, y

    def load_environment(self, checkpoint):
        Checkpoint.load_objects(
            {'model': self.model},
            checkpoint=checkpoint
        )
