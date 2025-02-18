import unittest

from nn.module import *


class TestEncoderDownLayer(unittest.TestCase):
    def test_encoder_down_layer(self):
        x = torch.rand(1, 3, 32, 32, 16)
        layer = EncoderDownLayer(
            16,
            (8, 8),
            (4, 4),
            8,
            2,
        )
        y = layer(x, [1, 3, 5])
        self.assertEqual(y.shape, (1, 3, 16, 16, 32))


class TestEncoderUpLayer(unittest.TestCase):
    def test_encoder_up_layer(self):
        x = torch.rand(1, 3, 16, 16, 16)
        h = torch.rand(1, 3, 32, 32, 8)
        layer = EncoderUpLayer(
            16,
            (8, 8),
            (4, 4),
            8,
            2,
        )
        y = layer(x, h, [1, 3, 5])
        self.assertEqual(y.shape, (1, 3, 32, 32, 8))


class TestEncoder(unittest.TestCase):
    def test_encoder(self):
        x = torch.rand(1, 3, 128, 128, 16)
        encoder = Encoder(16, 4, (4, 4), (2, 2), 4, 2)
        outputs = encoder(x, [0, 1, 2])
        print("outputs: ", [output.shape for output in outputs])
        self.assertEqual(outputs[-1].shape, (1, 3, 32, 32, 256))


class TestDecoderUpLayer(unittest.TestCase):
    def test_decoder_up_layer(self):
        x = torch.rand(1, 3, 16, 16, 32)
        h = torch.rand(1, 5, 16, 16, 32)
        layer = DecoderUpLayer(
            32,
            (8, 8),
            (4, 4),
            8,
            2,
        )
        y = layer(x, h, [[0, 2, 4, 6, 7], [1, 3, 5]])
        self.assertEqual(y.shape, (1, 3, 32, 32, 16))


class TestDecoder(unittest.TestCase):
    def test_decoder(self):
        x = torch.rand(1, 3, 128, 128, 16)
        encoder = Encoder(16, 4, (4, 4), (2, 2), 4, 2)
        outputs = encoder(x, [0, 1, 2])
        [print(output.shape) for output in outputs]
        decoder = Decoder(16, 4, (4, 4), (2, 2), 4, 2)
        B, M, H, W, C = outputs[0].shape
        noise = torch.rand(B, 1, H, W, C)
        y = decoder(noise, outputs, [
                    [0, 1, 2], [3]])
        self.assertEqual(y.shape, (1, 1, 128, 128, 16))
