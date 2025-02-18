from .module import *


class MultiContrastSwinTransformer(nn.Module):

    def __init__(self, dim, num_layers, window_size, shift_size, num_contrats, num_heads, patch_size=2):
        super(MultiContrastSwinTransformer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_contrats = num_contrats
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.image_encoding = MultiContrastImageEncoding(
            in_channels=1, out_channels=dim, num_contrats=num_contrats)
        self.image_decoding = MultiContrastImageDecoding(
            in_channels=dim, out_channels=1, num_contrats=num_contrats)

        self.encoder = Encoder(dim, num_layers, window_size,
                               shift_size, num_contrats, num_heads, patch_size)
        self.decoder = Decoder(dim, num_layers, window_size,
                               shift_size, num_contrats, num_heads, patch_size)

    def forward(self, x, selected_contrats):
        x = self.image_encoding(x, selected_contrats[0])
        encoded_features = self.encoder(x, selected_contrats[0])
        B, M, H, W, C = encoded_features[0].shape
        noise = torch.rand(B, len(selected_contrats[1]), H, W, C)
        decoded_features = self.decoder(
            noise, encoded_features, selected_contrats)
        y = self.image_decoding(decoded_features, selected_contrats[1])
        return y
