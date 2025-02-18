import unittest
import torch
from nn.block import *
from nn.utils import create_attention_mask


class TestWindowAttention(unittest.TestCase):
    def test_forward(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_forward2(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 3, 8, 8, 16)
        v = torch.randn(1, 3, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=1)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [2, 3, 5]])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_forward3(self):
        mask = torch.randn(16, 1, 3 * 2 * 2, 5 * 2 * 2)
        mask_ = create_attention_mask(3, 5, 8, 8, 2, (2, 2), (1, 1))
        self.assertTrue(mask.shape == mask_.shape,
                        f"{mask.shape}, {mask_.shape}")
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]], mask=mask_)
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_backward(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])

        # Backward pass
        y.sum().backward()


class TestMultiContrastEncoder(unittest.TestCase):
    def test_forward(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 8, 8, 16)

        # Create a MultiContrastEncoder module
        multi_contrast_encoder = MultiContrastEncoderBlock(
            dim=16, window_size=(4, 4), shift_size=(2, 2), num_contrasts=8, num_heads=2)

        # Forward pass
        y = multi_contrast_encoder(x, [2, 3, 5])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))


class TestMultiContrastDecoder(unittest.TestCase):
    def test_forward(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 8, 8, 16)
        features = torch.randn(1, 5, 8, 8, 16)

        # Create a MultiContrastDecoder module
        multi_contrast_decoder = MultiContrastDecoderBlock(
            dim=16, window_size=(4, 4), shift_size=(2, 2), num_contrasts=8, num_heads=2)

        # Forward pass
        y = multi_contrast_decoder(x, features, [[0, 1, 4, 6, 7], [2, 3, 5]])

        self.assertEqual(y.shape, (1, 3, 8, 8, 16))
