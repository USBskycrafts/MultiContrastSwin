import unittest

from multicontrast.nn.model import *
from multicontrast.nn.task import MultiModalityGeneration


class TestTransformer(unittest.TestCase):
    def test_transformer(self):
        transformer = MultiContrastSwinTransformer(
            dim=64, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrats=4, num_heads=4)
        x = torch.randn(32, 2, 128, 128, 1)
        selected_contrats = [[0, 3], [1, 2]]

        y = transformer(x, selected_contrats)
        self.assertEqual(y.shape, (32, 2, 128, 128, 1))

    def test_transformer2(self):
        transformer = MultiContrastSwinTransformer(
            dim=8, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrats=8, num_heads=4)
        x = torch.randn(32, 5, 128, 128, 1)
        selected_contrats = [[0, 3, 4, 7, 5], [1, 2, 6]]

        y = transformer(x, selected_contrats)
        self.assertEqual(y.shape, (32, 3, 128, 128, 1))


class TestMultiModalityGeneration(unittest.TestCase):
    def test_multimodality_generation(self):
        task = MultiModalityGeneration(
            dim=64, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrats=4, num_heads=4)

        x = torch.randn(32, 3, 128, 128, 1)
        selected_contrats = [0, 2, 3]
        y = task.predict(x, selected_contrats)
        loss = task.loss(x, selected_contrats, y)
        loss.backward()

        self.assertTrue(y.shape, torch.Size([32, 1, 128, 128, 1]))
