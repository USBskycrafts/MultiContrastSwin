import argparse
import sys
import unittest
import torch
from torch.utils.data import DataLoader

from multicontrast.dataset.tumor import MultiModalGenerationDataset
from multicontrast.nn.task import MultiModalityGeneration
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root_dir', type=str, required=True)
parser.add_argument('-m', '--modalities', nargs='+',
                    required=True)  # 移除 type=list
parser.add_argument('-s', '--slice_axis', type=int, default=2)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining  # 剩余参数给unittest


class TestMultiModalGenerationDataset(unittest.TestCase):
    def test_dataset_creation(self):
        dataset = MultiModalGenerationDataset(
            args.root_dir, args.modalities)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                                collate_fn=dataset.collate_fn, num_workers=0)
        for batch in dataloader:
            for i in range(batch['x'].shape[1]):
                sample = batch['x'][0, i, :, :, 0]
                plt.figure()
                plt.imshow(sample, cmap='gray', vmin=-1, vmax=1)
                plt.title(
                    f'Batch index: {batch["idx"][i]}, {batch["selected_contrasts"][i]}')
            plt.show()
    # @unittest.skip('Skip because time cost is too high')
    # def test_dataset_with_model(self):
    #     model = MultiModalityGeneration(dim=64, num_layers=4, window_size=(
    #         6, 5), shift_size=(3, 2), num_contrasts=len(args.modalities), num_heads=4)
    #     dataset = MultiModalGenerationDataset(
    #         args.root_dir, args.modalities)
    #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
    #                             collate_fn=dataset.collate_fn, num_workers=40)
    #     for batch in dataloader:
    #         model.eval()
    #         output = model(
    #             batch['x'], batch['selected_contrasts'], batch['generated_contrasts'])
    #         self.assertEqual(output.shape, batch['y'].shape)
    #         print(f"pass: {output.shape} == {batch['y'].shape}")
    #         model.train()
    #         loss = model(batch['x'], batch['selected_contrasts'],
    #                      batch['generated_contrasts'], batch['y'])
    #         self.assertTrue(loss.item() >= 0)
    #         print(f"pass: loss >= 0")


if __name__ == '__main__':
    unittest.main()
