import argparse
import os
from random import randint

from ignite.distributed.launcher import Parallel

from multicontrast.engine.model import MultiContrastGANGeneration


def train(local_rank, checkpoint=None):
    model = MultiContrastGANGeneration()
    model.train(checkpoint_path=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus if args.gpus else ''
    with Parallel('nccl', nproc_per_node=len(args.gpus.split(',')), nnodes=1, master_port=randint(20000, 30000)) as parallel:
        parallel.run(train, checkpoint=args.checkpoint)
