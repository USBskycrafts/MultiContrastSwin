import configparser
import os
from abc import ABCMeta, abstractmethod
from ast import literal_eval

import torch
from ignite.distributed import auto_dataloader, auto_model, auto_optim

from multicontrast.dataset.tumor import MultiModalGenerationDataset
from multicontrast.engine.trainer import SupervisedTrainer
from multicontrast.engine.validator import SupervisedValidator
from multicontrast.nn.task import MultiModalityGeneration
from multicontrast.utils import DEFAULT_CFG_PATH, ROOT


class Model(metaclass=ABCMeta):
    def __init__(self, config=DEFAULT_CFG_PATH):
        # Load main config
        dataset_cfg_path = ROOT / 'config/dataset.ini'
        self.config = configparser.ConfigParser()
        self.config.read([config, dataset_cfg_path])
        print(f'Loading config from {config}, {dataset_cfg_path}')

    def train(self, checkpoint_path=None):
        self._train(checkpoint_path)

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
        # dim, num_layers, window_size, shift_size, num_contrats, num_heads
        config = self.config
        model_config = {
            "dim": config.getint('model', 'dim'),
            "num_layers": config.getint('model', 'num_layers'),
            "window_size": literal_eval(config.get('model', 'window_size')),
            "shift_size": literal_eval(config.get('model', 'shift_size')),
            "num_contrats": config.getint('model', 'num_contrats'),
            "num_heads": config.getint('model', 'num_heads')
        }
        self.model = MultiModalityGeneration(**model_config)
        self.model = auto_model(self.model, find_unused_parameters=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.getfloat('train', 'learning_rate'))
        self.optimizer = auto_optim(self.optimizer)
        self.training_dataset = MultiModalGenerationDataset(
            root_dir=config.get('train', 'root_dir'),
            modalities=config.get('dataset', 'modalities').split(','),
        )

    def _train(self, checkpoint_path=None):
        self.trainer = SupervisedTrainer(self.model,
                                         self.optimizer)
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
        self._prepare_eval_data(True)
        self.validator.load_environment(checkpoint_path)
        self.validator.validate(self.data_loader)

    def _prepare_eval_data(self, save_image=False):
        self.validator = SupervisedValidator(self.model,
                                             output_dir=self.config.get(
                                                 'test', 'output_dir'),
                                             save_images=save_image)
        self.validator.load_environment(self.config['checkpoint']['path'])

        if self.validation_dataset is None:
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
            batch_size=self.config.getint('val', 'batch_size'),
            shuffle=False,
            num_workers=self.config.getint('val', 'num_workers'),
            pin_memory=True,
            drop_last=False,
            collate_fn=self.validation_dataset.collate_fn,
        )
