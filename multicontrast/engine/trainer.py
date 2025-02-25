from abc import ABCMeta, abstractmethod

import ignite.distributed as distributed
import torch
from ignite.distributed import auto_dataloader, auto_model, auto_optim
from ignite.engine import Engine, Events
from ignite.handlers import (Checkpoint, ProgressBar, TensorboardLogger,
                             TerminateOnNan, global_step_from_engine)
from ignite.metrics import RunningAverage
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from multicontrast.nn.task import MultiModalityGeneration
from multicontrast.utils.metrics import StablePSNR as PSNR
from multicontrast.utils.metrics import StableSSIM as SSIM


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self):
        self.engine = Engine(lambda engine, batch: self._train_step(batch))

    def train(self, data_loader, num_epochs, *args, **kwargs):
        """_summary_

        Args:
            num_epochs,
            every_save,
            save_handler,
        """
        self._set_checkpoint(*args, **kwargs)
        self.register_events(Events.ITERATION_COMPLETED,
                             TerminateOnNan())
        RunningAverage(output_transform=lambda x: x).attach(
            self.engine, "loss")
        ProgressBar(persist=False, desc="Training").attach(
            self.engine, ["loss"])
        self.engine.run(data_loader, num_epochs)

    @abstractmethod
    def _train_step(self, batch):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    def validate(self, data_loader):
        validator = Engine(
            lambda engine, batch: self._validate_step(batch))
        psnr = PSNR(data_range=1, device=distributed.device(),
                    output_transform=lambda y: (y[0].squeeze(-1), y[1].squeeze(-1)))
        ssim = SSIM(data_range=1, device=distributed.device(),
                    output_transform=lambda y: (y[0].squeeze(-1), y[1].squeeze(-1)))
        psnr.attach(validator, "psnr")
        ssim.attach(validator, "ssim")
        if self.log_dir is not None:
            logger = TensorboardLogger(self.log_dir)
            logger.attach_output_handler(
                validator,
                event_name=Events.ITERATION_COMPLETED,
                tag="validation",
                metric_names=["psnr", "ssim"],
                global_step_transform=global_step_from_engine(self.engine)
            )
        validator.run(data_loader, 1)
        self.engine.state.metrics["psnr"] = validator.state.metrics["psnr"]
        self.engine.state.metrics["ssim"] = validator.state.metrics["ssim"]
        return validator.state.metrics["psnr"], validator.state.metrics["ssim"]

    @abstractmethod
    def _validate_step(self, batch):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    def register_events(self, event_name, handler):
        self.engine.add_event_handler(event_name, handler)

    def register_tensorboard(self, log_dir):
        self.log_dir = log_dir
        tb_logger = TensorboardLogger(log_dir)
        tb_logger.attach_output_handler(
            self.engine,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"loss": loss}
        )

    def register_validation(self, data_loader, every_epochs):
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=every_epochs),
            self.validate,
            data_loader
        )

    @abstractmethod
    def _set_checkpoint(self, every_save, *args, **kwargs):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    @abstractmethod
    def load_environment(self, checkpoint):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()

    def _train_step(self, batch):
        self.model.train()
        x = batch['x'].to(distributed.device())
        y = batch['y'].to(distributed.device())
        selected_contrasts = batch['selected_contrasts']
        generated_contrats = batch['generated_contrasts']
        with autocast():
            loss = self.model(x, selected_contrasts,
                              generated_contrats, y).mean()
        scaler_loss = self.scaler.scale(loss)
        if isinstance(scaler_loss, torch.Tensor):
            scaler_loss.backward()
        else:
            raise ValueError("Loss must be a tensor")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def _set_checkpoint(self, every_save, *args, **kwargs):
        self.register_events(Events.EPOCH_COMPLETED(every=every_save),
                             Checkpoint({'model': self.model,
                                         'optimizer': self.optimizer,
                                         'scaler': self.scaler},
                                        *args,
                                        score_name='psnrxssim',
                                        score_function=lambda engine: engine.state.metrics['psnr'] *
                                        engine.state.metrics['ssim'],
                                        n_saved=10,
                                        **kwargs))

    def load_environment(self, checkpoint):
        Checkpoint.load_objects(
            {'model': self.model,
             'optimizer': self.optimizer,
             'scaler': self.scaler},
            checkpoint=checkpoint
        )
    # def validate(self, data_loader):
    #     data_loader = auto_dataloader(data_loader)
    #     validator = Engine(
    #         lambda engine, batch: self._validate_step(batch))
    #     if self.log_dir is not None:
    #         logger = TensorboardLogger(self.log_dir)
    #         logger.attach_output_handler(
    #             validator,
    #             event_name=Events.ITERATION_COMPLETED,
    #             tag="validation",
    #             metrics={"psnr", "ssim"},
    #             global_step_tranforms=global_step_from_engine(self.engine)
    #         )
    #     psnr = RunningAverage(PSNR(data_range=1, device=distributed.device()))
    #     ssim = RunningAverage(SSIM(data_range=1, device=distributed.device()))
    #     psnr.attach(validator, "psnr")
    #     ssim.attach(validator, "ssim")
    #     validator.run(data_loader, 1)
    #     return validator.state.metrics["psnr"], validator.state.metrics["ssim"]

    def _validate_step(self, batch):
        self.model.eval()
        x = batch['x'].to(distributed.device())
        y = batch['y'].to(distributed.device())
        selected_contrasts = batch['selected_contrasts']
        generated_contrats = batch['generated_contrasts']
        with torch.no_grad():
            with autocast():
                pred = self.model(x, selected_contrasts, generated_contrats)
        return pred, y

    # def train(self, num_epochs, every_save, *args, **kwargs):
    #     self.register_events(Events.EPOCH_COMPLETED(every=every_save),
    #                          ModelCheckpoint(*args, **kwargs))
    #     self.engine.run(self.data_loader, num_epochs)

    # def register_events(self, event_name, handler):
    #     self.engine.add_event_handler(event_name, handler)

    # def register_tensorboard(self, log_dir):
    #     self.log_dir = log_dir
    #     tb_logger = TensorboardLogger(log_dir)
    #     tb_logger.attach_output_handler(
    #         self.engine,
    #         event_name=Events.ITERATION_COMPLETED,
    #         tag="training",
    #         output_transform=lambda loss: {"loss": loss}
    #     )

    # def register_validation(self, data_loader, every_epochs):
    #     self.engine.add_event_handler(
    #         Events.EPOCH_COMPLETED(every=every_epochs),
    #         self.validate,
    #         data_loader
    #     )
