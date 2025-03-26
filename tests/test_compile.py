import unittest
import torch

from multicontrast.nn.task import *


class TestCompiledable(unittest.TestCase):

    def setUp(self):
        task = MultiModalityGeneration(
            dim=16, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrasts=4, num_heads=4)
        task.train()
        task = torch.compile(task)
        print('Task compiled successfully!')
        self.task = task

    def test_compiledable(self):
        task = self.task
        x = torch.randn(2, 1, 128, 128, 1)
        y = torch.randn(2, 1, 128, 128, 1)
        selected_contrasts = [0]
        generated_contrasts = [1]

        loss, y = task(x, selected_contrasts, generated_contrasts, y)
        loss.backward()

        self.assertTrue(y.shape, torch.Size([2, 1, 128, 128, 1]))
        print('Test passed!')
