from abc import ABCMeta, abstractmethod

import ignite.distributed as distributed
import torch
from ignite.engine import Engine, Events
from ignite.handlers import (Checkpoint, ProgressBar, TensorboardLogger,
                             TerminateOnNan, global_step_from_engine)
from ignite.metrics import RunningAverage
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from multicontrast.utils.metrics import StablePSNR as PSNR
from multicontrast.utils.metrics import StableSSIM as SSIM
from multicontrast.utils.metrics import range_transform


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
                    output_transform=lambda y: range_transform((y[0].squeeze(-1), y[1].squeeze(-1))))
        ssim = SSIM(data_range=1, device=distributed.device(),
                    output_transform=lambda y: range_transform((y[0].squeeze(-1), y[1].squeeze(-1))))
        psnr.attach(validator, "psnr")
        ssim.attach(validator, "ssim")
        if self.log_dir is not None:
            logger = TensorboardLogger(self.log_dir)
            logger.attach_output_handler(
                validator,
                event_name=Events.COMPLETED,
                tag="validation",
                metric_names=["psnr", "ssim"],
                global_step_transform=global_step_from_engine(self.engine)
            )
        validator.run(data_loader, 1)
        validator.add_event_handler(
            Events.COMPLETED,
            lambda *_: print(f"""PSNR: {validator.state.metrics["psnr"]}\n
                             SSIM: {validator.state.metrics["ssim"]}"""),
        )
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


class GANTrainer(BaseTrainer):
    def __init__(self,
                 generator,
                 discriminator,
                 g_optim,
                 d_optim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.d_optim = d_optim

        self.g_scaler = GradScaler()
        self.d_scaler = GradScaler()

    def _train_step(self, batch):
        self.generator.eval()
        self.discriminator.train()
        fake = self.generator(batch['x'],
                              batch['selected_contrasts'],
                              batch['generated_contrasts'])
        real = batch['y']
        real_label = torch.tensor(1.0).to(distributed.device())
        fake_label = torch.tensor(0.0).to(distributed.device())
        with autocast():
            real_loss = self.discriminator(
                real, batch['generated_contrasts'], real_label)
            fake_loss = self.discriminator(
                fake.detach(), batch['generated_contrasts'], fake_label)
        d_loss = (real_loss + fake_loss) / 2
        scaler_loss = self.d_scaler.scale(d_loss)
        if not isinstance(scaler_loss, torch.Tensor):
            raise RuntimeError(
                f"Expected scaler_loss to be a tensor, but got {type(scaler_loss)}"
            )
        scaler_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.d_scaler.step(self.d_optim)
        self.d_scaler.update()
        self.d_optim.zero_grad()

        self.generator.train()
        with autocast():
            g_l1_loss = self.generator(
                batch['x'],
                batch['selected_contrasts'],
                batch['generated_contrasts'],
                real
            )
            g_against_loss = self.discriminator(
                fake,
                batch['generated_contrasts'],
                real_label
            )
        g_loss = g_l1_loss + g_against_loss
        scaled_loss = self.g_scaler.scale(g_loss)
        if not isinstance(scaled_loss, torch.Tensor):
            raise RuntimeError(
                f"Expected scaled_loss to be a tensor, but got {type(scaled_loss)}"
            )
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_scaler.step(self.g_optim)
        self.g_scaler.update()
        self.g_optim.zero_grad()

        return torch.tensor([g_loss.item(), d_loss.item()])

    def _validate_step(self, batch):
        self.generator.eval()
        with torch.no_grad():
            with autocast():
                fake = self.generator(
                    batch['x'],
                    batch['selected_contrasts'],
                    batch['generated_contrasts'],
                )
        return fake, batch['y']

    def load_environment(self, checkpoint):
        Checkpoint.load_objects({
            'generator': self.generator,
            'discriminator': self.discriminator,
            'g_optim': self.g_optim,
            'd_optim': self.d_optim,
            'g_scaler': self.g_scaler,
            'd_scaler': self.d_scaler,
        }, checkpoint)

    def _set_checkpoint(self, every_save, *args, **kwargs):
        self.register_events(Events.EPOCH_COMPLETED(every=every_save),
                             Checkpoint({
                                 'generator': self.generator,
                                 'discriminator': self.discriminator,
                                 'g_optim': self.g_optim,
                                 'd_optim': self.d_optim,
                                 'g_scaler': self.g_scaler,
                                 'd_scaler': self.d_scaler,
                             },
            *args,
            score_name='psnrxssim',
            score_function=lambda engine: engine.state.metrics['psnr'] *
            engine.state.metrics['ssim'],
            n_saved=10,
            **kwargs))
