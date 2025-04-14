import argparse
import os
import random

import numpy as np
import torch
from ignite.distributed.launcher import Parallel

from multicontrast.engine.model import MultiContrastGeneration


def train(local_rank, checkpoint_path):
    model = MultiContrastGeneration()
    checkpoint = torch.load(checkpoint_path)
    model.predict(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, args.gpus)) if args.gpus else ''
    random.seed(args.seed)
    np.random.seed(args.seed)  # for yolov5-mosaic
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with Parallel('nccl', len(args.gpus), master_port=29600) as parallel:
        parallel.run(train, args.checkpoint_path)
