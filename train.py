import os

from ignite.distributed.launcher import Parallel

from multicontrast.engine.model import MultiContrastGeneration


def train(local_rank):
    model = MultiContrastGeneration()
    model.train()


if __name__ == "__main__":
    with Parallel('nccl', len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))) as parallel:
        parallel.run(train)
