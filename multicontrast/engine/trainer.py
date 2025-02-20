from abc import ABCMeta, abstractmethod

import ignite.distributed as distributed
import torch
from ignite.distributed import auto_dataloader, auto_model, auto_optim
from ignite.engine import Engine, Events
from ignite.handlers import (Checkpoint, TensorboardLogger, TerminateOnNan,
                             global_step_from_engine)
from ignite.metrics import PSNR, SSIM, RunningAverage
from nn.task import MultiModalityGeneration
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, data_loader):
        self.data_loader = auto_dataloader(data_loader)

        self.engine = Engine(lambda engine, batch: self._train_step(batch))

    def train(self, num_epochs, *args, **kwargs):
        self._set_checkpoint(*args, **kwargs)
        self.register_events(Events.ITERATION_COMPLETED,
                             TerminateOnNan())
        self.engine.run(self.data_loader, num_epochs)

    @abstractmethod
    def _train_step(self, batch):
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    def validate(self, data_loader):
        data_loader = auto_dataloader(data_loader)
        validator = Engine(
            lambda engine, batch: self._validate_step(batch))
        if self.log_dir is not None:
            logger = TensorboardLogger(self.log_dir)
            logger.attach_output_handler(
                validator,
                event_name=Events.ITERATION_COMPLETED,
                tag="validation",
                metrics={"psnr", "ssim"},
                global_step_tranforms=global_step_from_engine(self.engine)
            )
        psnr = RunningAverage(PSNR(data_range=1, device=distributed.device()))
        ssim = RunningAverage(SSIM(data_range=1, device=distributed.device()))
        psnr.attach(validator, "psnr")
        ssim.attach(validator, "ssim")
        validator.run(data_loader, 1)
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
    def __init__(self, model: MultiModalityGeneration, optimizer, data_loader):
        super().__init__(data_loader)
        self.model = auto_model(model)
        self.optimizer = auto_optim(optimizer)
        self.scaler = GradScaler()

    def _train_step(self, batch):
        self.model.train()
        x = batch['x']
        y = batch['y']
        selected_contrasts = batch['selected_contrasts']
        generated_contrats = batch['generated_contrasts']
        with autocast():
            loss = self.model(x, selected_contrasts, generated_contrats, y)
        loss = self.scaler.scale(loss)
        if isinstance(loss, torch.Tensor):
            loss.backward()
        else:
            raise ValueError("Loss must be a tensor")
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
        x = batch['x']
        y = batch['y']
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
