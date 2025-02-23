import configparser
import os
from abc import ABCMeta, abstractmethod
from ast import literal_eval

import ignite.distributed.launcher as launcher
from ignite.distributed import auto_dataloader
import torch
from torchvision.datasets import CIFAR10


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

        # Init distributed config
        self.distributed_config = {
            'backend': self.config['distributed']['backend'],
            'nproc_per_node': int(self.config['distributed']['nproc_per_node']),
            'nnodes': int(os.getenv('CUDA_VISIBLE_DEVICES', '1')),
            'node_rank': int(self.config['distributed']['node_rank']),
            'master_addr': self.config['distributed']['master_addr'],
            'master_port': int(self.config['distributed']['master_port'])
        }

    def train(self):
        # init the distributed environ

        with launcher.Parallel(**self.distributed_config):
            self._train()

    @abstractmethod
    def _train(self):
        pass

    def evaluate(self):
        with launcher.Parallel(**self.distributed_config):
            self._evaluate()

    @abstractmethod
    def _evaluate(self):
        pass

    def predict(self):
        with launcher.Parallel(**self.distributed_config):
            self._predict()

    @abstractmethod
    def _predict(self):
        pass


class MultiContrastGneration(Model):
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
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.getfloat('train', 'learning_rate'))
        self.training_dataset = MultiModalGenerationDataset(
            root_dir=config.get('train', 'root_dir'),
            modalities=config.get('dataset', 'modalities').split(','),
        )

        self.data_loader = auto_dataloader(
            self.training_dataset,
            batch_size=config.getint('train', 'batch_size'),
            shuffle=True,
            num_workers=config.getint('train', 'num_workers'),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.training_dataset.collate_fn,
        )

        self.trainer = SupervisedTrainer(self.model,
                                         self.optimizer)

    def _train(self):
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
            data_loader,
            self.config.getint('train', 'num_epochs'),
            every_save=self.config.getint('train', 'every_save'),
            save_handler=self.config.get('train', 'log_dir')
        )

    def _evaluate(self):
        self._prepare_eval_data(False)
        self.validator.validate(self.data_loader)

    def _predict(self):
        self._prepare_eval_data(True)
        self.validator.validate(self.data_loader)

    def _prepare_eval_data(self, save_image=False):
        self.validator = SupervisedValidator(self.model,
                                             output_dir=self.config.get(
                                                 'test', 'output_dir'),
                                             save_images=save_image)
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
