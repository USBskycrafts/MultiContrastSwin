import argparse
from ast import arg
import sys
import unittest

from torch.utils.data import DataLoader

from multicontrast.dataset.tumor import MultiModalMRIDataset


class TestMultiModalGenerationDataset(unittest.TestCase):

    def test_dataset_creation(self, args):
        dataset = MultiModalMRIDataset(
            args.root_dir, args.modalities, args.slice_axis)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                                collate_fn=dataset.collate_fn, num_workers=40)
        for batch in dataloader:
            print(batch['x'].shape)
            print(batch['y'].shape)
            print(batch['selected_contrasts'])
            print(batch['generated_contrasts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', type=str, required=True)
    parser.add_argument('-m', '--modalities', nargs='+',
                        required=True)  # 移除 type=list
    parser.add_argument('-s', '--slice_axis', type=int, default=2)
    args = parser.parse_args()

    test = TestMultiModalGenerationDataset()
    test.test_dataset_creation(args)
