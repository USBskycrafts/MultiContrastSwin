import argparse
import os

import torch

from ignite.distributed.launcher import Parallel

from multicontrast.engine.model import MultiContrastGeneration, MultiContrastGANGeneration


def train(local_rank, checkpoint_path):
    # model = MultiContrastGeneration()
    model = MultiContrastGANGeneration()
    checkpoint = torch.load(checkpoint_path)
    model.predict(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, args.gpus)) if args.gpus else ''
    with Parallel('nccl', len(args.gpus), master_port=29600) as parallel:
        parallel.run(train, args.checkpoint_path)
