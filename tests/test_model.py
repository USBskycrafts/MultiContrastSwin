import unittest

from multicontrast.nn.model import *
from multicontrast.nn.task import MultiModalityGeneration


class TestTransformer(unittest.TestCase):
    def test_transformer(self):
        transformer = MultiContrastSwinTransformer(
            dim=16, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrasts=4, num_heads=4)
        x = torch.randn(32, 2, 128, 128, 1)
        selected_contrats = [[0, 3], [1, 2]]

        y = transformer(x, selected_contrats)
        self.assertEqual(y.shape, (32, 2, 128, 128, 1))

    def test_transformer2(self):
        transformer = MultiContrastSwinTransformer(
            dim=8, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrasts=8, num_heads=4)
        x = torch.randn(32, 5, 128, 128, 1)
        selected_contrats = [[0, 3, 4, 7, 5], [1, 2, 6]]

        y = transformer(x, selected_contrats)
        self.assertEqual(y.shape, (32, 3, 128, 128, 1))


class TestMultiModalityGeneration(unittest.TestCase):
    def test_multimodality_generation(self):
        task = MultiModalityGeneration(
            dim=16, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrasts=4, num_heads=4)

        x = torch.randn(32, 1, 128, 128, 1)
        selected_contrasts = [0]
        generated_contrasts = [1]
        y = task.predict(x, selected_contrasts, generated_contrasts)
        loss = task.loss(x, selected_contrasts, generated_contrasts, y)
        loss.backward()

        self.assertTrue(y.shape, torch.Size([32, 1, 128, 128, 1]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    @unittest.skipIf(torch.cuda.device_count() < 2, "Less than 2 CUDA devices available")
    # skip when cuda does not have enough memory
    @unittest.skipIf(torch.cuda.memory_allocated() > 1024 * 1024 * 1024, "CUDA memory allocated is too high")
    def test_model_on_cuda(self):
        if torch.cuda.is_available():
            task = MultiModalityGeneration(
                dim=4, num_layers=4, window_size=(4, 4), shift_size=(2, 2), num_contrasts=4, num_heads=4).cuda()
            x = torch.randn(32, 1, 128, 128, 1).cuda()
            selected_contrasts = [0]
            generated_contrasts = [1]
            y = task.predict(x, selected_contrasts, generated_contrasts)
            loss = task.loss(x, selected_contrasts, generated_contrasts, y)
            loss.backward()
            self.assertTrue(y.shape, torch.Size([32, 1, 128, 128, 1]))
