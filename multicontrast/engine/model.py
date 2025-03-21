import configparser
import os
from abc import ABCMeta, abstractmethod
from ast import literal_eval
from typing import Union

import torch
from ignite.distributed import auto_dataloader, auto_model, auto_optim
from ignite.handlers import CosineAnnealingScheduler,  create_lr_scheduler_with_warmup
from ignite.engine import Events
from multicontrast.dataset.tumor import MultiModalGenerationDataset
from multicontrast.engine.trainer import GANTrainer, SupervisedTrainer
from multicontrast.engine.validator import SupervisedValidator
from multicontrast.nn.task import (MultiContrastDiscrimination,
                                   MultiModalityGeneration)
from multicontrast.utils import DEFAULT_CFG_PATH, ROOT


class Model(metaclass=ABCMeta):
    def __init__(self, config=DEFAULT_CFG_PATH):
        # Load main config
        dataset_cfg_path = ROOT / 'config/dataset.ini'
        self.config = configparser.ConfigParser()
        self.config.read([config, dataset_cfg_path])
        torch.set_float32_matmul_precision('high')

    def train(self, checkpoint_path: Union[str, None] = None):
        if checkpoint_path is None:
            self._train(checkpoint_path)
        else:
            self._train(torch.load(checkpoint_path, map_location='cpu'))

    @abstractmethod
    def _train(self, checkpoint_path):
        pass

    # evaluate和predict同理
    def evaluate(self, checkpoint_path):
        self._evaluate(checkpoint_path)

    @abstractmethod
    def _evaluate(self, checkpoint_path):
        pass

    def predict(self, checkpoint_path):
        self._predict(checkpoint_path)

    @abstractmethod
    def _predict(self, checkpoint_path):
        pass


class MultiContrastGeneration(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init the model, dataset, etc
        # attr:
        # dim, num_layers, window_size, shift_size, num_contrasts, num_heads
        config = self.config
        model_config = {
            "dim": config.getint('model', 'dim'),
            "num_layers": config.getint('model', 'num_layers'),
            "window_size": literal_eval(config.get('model', 'window_size')),
            "shift_size": literal_eval(config.get('model', 'shift_size')),
            "num_contrasts": config.getint('model', 'num_contrasts'),
            "num_heads": config.getint('model', 'num_heads')
        }
        self.model = MultiModalityGeneration(**model_config)
        self.model = auto_model(self.model, find_unused_parameters=True)
        self.learning_rate = config.getfloat('train', 'learning_rate')
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate)
        self.optimizer = auto_optim(self.optimizer)
        self.training_dataset = MultiModalGenerationDataset(
            root_dir=config.get('train', 'root_dir'),
            modalities=config.get('dataset', 'modalities').split(','),
        )

    def _train(self, checkpoint_path=None):
        self.trainer = SupervisedTrainer(self.model,
                                         self.optimizer)

        scheduler = CosineAnnealingScheduler(self.optimizer, "lr",
                                             self.learning_rate, 0.1 * self.learning_rate, 1000,
                                             cycle_mult=1.1)
        scheduler = create_lr_scheduler_with_warmup(
            scheduler,
            warmup_start_value=0.0,
            warmup_duration=200,
            warmup_end_value=self.learning_rate,
        )
        self.trainer.register_events(Events.ITERATION_STARTED, scheduler)
        self.data_loader = auto_dataloader(
            self.training_dataset,
            batch_size=self.config.getint('train', 'batch_size'),
            shuffle=True,
            num_workers=self.config.getint('train', 'num_workers'),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.training_dataset.collate_fn,
        )

        if checkpoint_path is not None:
            self.trainer.load_environment(checkpoint_path)

        self.trainer.register_tensorboard(
            log_dir=self.config.get('train', 'log_dir'),
        )

        self.validation_dataset = MultiModalGenerationDataset(
            root_dir=self.config.get('val', 'root_dir'),
            modalities=['t1', 't2', 'flair', 't1ce'],
        )

        data_loader = auto_dataloader(self.validation_dataset,
                                      batch_size=self.config.getint(
                                          'val', 'batch_size'),
                                      shuffle=False,
                                      num_workers=self.config.getint(
                                          'val', 'num_workers'),
                                      pin_memory=True,
                                      drop_last=False,
                                      collate_fn=self.validation_dataset.collate_fn,)

        self.trainer.register_validation(
            data_loader=data_loader,
            every_epochs=self.config.getint('val', 'every_epochs'),
        )

        self.trainer.train(
            self.data_loader,
            self.config.getint('train', 'num_epochs'),
            every_save=self.config.getint('train', 'every_save'),
            save_handler=self.config.get('train', 'log_dir')
        )

    def _evaluate(self, checkpoint_path):
        self._prepare_eval_data(False)
        self.validator.load_environment(checkpoint_path)
        self.validator.validate(self.data_loader)

    def _predict(self, checkpoint_path):
        self._prepare_eval_data(checkpoint_path, True)
        self.validator.load_environment(checkpoint_path)
        self.validator.validate(self.data_loader)

    def _prepare_eval_data(self, checkpoint_path, save_image=False):
        self.validator = SupervisedValidator(self.model,
                                             output_dir=self.config.get(
                                                 'test', 'output_dir'),
                                             save_images=save_image)
        self.validator.load_environment(checkpoint_path)

        if getattr(self, 'validation_dataset', None) is None:
            self.validation_dataset = MultiModalGenerationDataset(
                root_dir=self.config.get('test', 'root_dir'),
                modalities=['t1', 't2', 'flair', 't1ce'],
                selected_contrasts=literal_eval(
                    self.config.get('test', 'selected_contrasts')),
                generated_contrasts=literal_eval(
                    self.config.get('test', 'generated_contrasts')),
            )

        self.data_loader = auto_dataloader(
            self.validation_dataset,
            batch_size=self.config.getint('test', 'batch_size'),
            shuffle=False,
            num_workers=self.config.getint('test', 'num_workers'),
            pin_memory=True,
            drop_last=False,
            collate_fn=self.validation_dataset.collate_fn,
        )


class MultiContrastGANGeneration(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init the model, dataset, etc
        # attr:
        # dim, num_layers, window_size, shift_size, num_contrasts, num_heads
        config = self.config
        model_config = {
            "dim": config.getint('model', 'dim'),
            "num_layers": config.getint('model', 'num_layers'),
            "window_size": literal_eval(config.get('model', 'window_size')),
            "shift_size": literal_eval(config.get('model', 'shift_size')),
            "num_contrasts": config.getint('model', 'num_contrasts'),
            "num_heads": config.getint('model', 'num_heads')
        }
        self.generator = MultiModalityGeneration(**model_config)
        self.generator = auto_model(
            self.generator, find_unused_parameters=True)
        self.g_lr = config.getfloat('train', 'g_lr')
        self.g_optim = torch.optim.AdamW(self.generator.parameters(),
                                         lr=self.g_lr)
        self.g_optim = auto_optim(self.g_optim)
        # ----------------------------------------------------------------------------
        self.discriminator = MultiContrastDiscrimination(**model_config)
        self.discriminator = auto_model(
            self.discriminator, find_unused_parameters=True)
        self.d_lr = config.getfloat('train', 'd_lr')
        self.d_optim = torch.optim.AdamW(self.discriminator.parameters(),
                                         self.d_lr)
        self.d_optim = auto_optim(self.d_optim)

        self.training_dataset = MultiModalGenerationDataset(
            root_dir=config.get('train', 'root_dir'),
            modalities=config.get('dataset', 'modalities').split(','),
        )

    def _train(self, checkpoint_path=None):
        self.trainer = GANTrainer(self.generator,
                                  self.discriminator,
                                  self.g_optim,
                                  self.d_optim)
        g_sche = CosineAnnealingScheduler(self.g_optim, "lr",
                                          self.g_lr, 0.1 * self.g_lr, 1000,
                                          cycle_mult=1.1)
        g_sche = create_lr_scheduler_with_warmup(
            g_sche,
            warmup_start_value=0.0,
            warmup_duration=200,
            warmup_end_value=self.g_lr,
        )
        self.trainer.register_events(Events.ITERATION_STARTED, g_sche)
        # --------------------------------------------------------------------
        d_sche = CosineAnnealingScheduler(self.d_optim, "lr",
                                          self.d_lr, 0.1 * self.d_lr, 1000,
                                          cycle_mult=1.1)
        d_sche = create_lr_scheduler_with_warmup(
            d_sche,
            warmup_start_value=0.0,
            warmup_duration=200,
            warmup_end_value=self.g_lr,
        )
        self.trainer.register_events(Events.ITERATION_STARTED, d_sche)

        self.data_loader = auto_dataloader(
            self.training_dataset,
            batch_size=self.config.getint('train', 'batch_size'),
            shuffle=True,
            num_workers=self.config.getint('train', 'num_workers'),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.training_dataset.collate_fn,
        )

        if checkpoint_path is not None:
            self.trainer.load_environment(checkpoint_path)

        self.trainer.register_tensorboard(
            log_dir=self.config.get('train', 'log_dir'),
        )

        self.validation_dataset = MultiModalGenerationDataset(
            root_dir=self.config.get('val', 'root_dir'),
            modalities=['t1', 't2', 'flair', 't1ce'],
        )

        data_loader = auto_dataloader(self.validation_dataset,
                                      batch_size=self.config.getint(
                                          'val', 'batch_size'),
                                      shuffle=False,
                                      num_workers=self.config.getint(
                                          'val', 'num_workers'),
                                      pin_memory=True,
                                      drop_last=False,
                                      collate_fn=self.validation_dataset.collate_fn,)

        self.trainer.register_validation(
            data_loader=data_loader,
            every_epochs=self.config.getint('val', 'every_epochs'),
        )

        self.trainer.train(
            self.data_loader,
            self.config.getint('train', 'num_epochs'),
            every_save=self.config.getint('train', 'every_save'),
            save_handler=self.config.get('train', 'log_dir')
        )

    def _evaluate(self, checkpoint_path):
        self._prepare_eval_data(False)
        self.validator.load_environment(checkpoint_path)
        self.validator.validate(self.data_loader)

    def _predict(self, checkpoint_path):
        self._prepare_eval_data(checkpoint_path, True)
        self.validator.load_environment(checkpoint_path)
        self.validator.validate(self.data_loader)

    def _prepare_eval_data(self, checkpoint_path, save_image=False):
        self.validator = SupervisedValidator(self.generator,
                                             output_dir=self.config.get(
                                                 'test', 'output_dir'),
                                             save_images=save_image)
        self.validator.load_environment(checkpoint_path)

        if getattr(self, 'validation_dataset', None) is None:
            self.validation_dataset = MultiModalGenerationDataset(
                root_dir=self.config.get('test', 'root_dir'),
                modalities=['t1', 't2', 'flair', 't1ce'],
                selected_contrasts=literal_eval(
                    self.config.get('test', 'selected_contrasts')),
                generated_contrasts=literal_eval(
                    self.config.get('test', 'generated_contrasts')),
            )

        self.data_loader = auto_dataloader(
            self.validation_dataset,
            batch_size=self.config.getint('test', 'batch_size'),
            shuffle=False,
            num_workers=self.config.getint('test', 'num_workers'),
            pin_memory=True,
            drop_last=False,
            collate_fn=self.validation_dataset.collate_fn,
        )
