from .block import *


class EncoderDownLayer(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrats, num_heads, patch_size=2, reduction=True):
        super(EncoderDownLayer, self).__init__()
        self.patches = PatchPartition(dim, patch_size, reduction)
        if reduction:
            dim *= patch_size
        else:
            dim *= (patch_size ** 2)
        self.layer = nn.Sequential(
            MultiContrastEncoderBlock(dim,
                                      window_size, (0, 0),
                                      num_contrats, num_heads),
            MultiContrastEncoderBlock(dim,
                                      window_size, shift_size,
                                      num_contrats, num_heads)
        )

    def forward(self, x, selected_contrats):
        h = self.patches(x)
        for layer in self.layer:
            h = layer(h, selected_contrats)
        return h


class EncoderUpLayer(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrats, num_heads, patch_size=2, reduction=True):
        super(EncoderUpLayer, self).__init__()
        self.expands = PatchExpansion(dim, patch_size, reduction)
        self.layer = nn.Sequential(
            MultiContrastEncoderBlock(dim // patch_size,
                                      window_size, (0, 0),
                                      num_contrats, num_heads),
            MultiContrastEncoderBlock(dim // patch_size,
                                      window_size, shift_size,
                                      num_contrats, num_heads),
        )

        self.compress = nn.Linear(2 * dim // (patch_size), dim // patch_size)

    def forward(self, x, skiped_features, selected_contrats):
        x = self.expands(x)
        x = torch.cat([x, skiped_features], dim=-1)
        x = self.compress(x)
        for layer in self.layer:
            x = layer(x, selected_contrats)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, num_layers, window_size, shift_size, num_contrats, num_heads, patch_size=2):
        super(Encoder, self).__init__()
        # self.patches = PatchPartition(dim, patch_size * 2, False)
        self.patches = EncoderDownLayer(
            dim, window_size, shift_size, num_contrats, num_heads, patch_size * 2, False)

        self.down_layers = nn.ModuleList([
            EncoderDownLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                             num_contrats, num_heads, patch_size)
            for i in range(num_layers)
        ])

        self.bottleneck = nn.Sequential(
            MultiContrastEncoderBlock(dim * (1 << num_layers) * patch_size ** 4,
                                      window_size, (0, 0),
                                      num_contrats, num_heads),
            MultiContrastEncoderBlock(dim * (1 << num_layers) * patch_size ** 4,
                                      window_size, shift_size,
                                      num_contrats, num_heads)
        )

        self.up_layers = nn.ModuleList([
            EncoderUpLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                           num_contrats, num_heads, patch_size)
            for i in range(num_layers, 0, -1)
        ])

    def forward(self, x, selected_contrats):
        skiped_features = []
        output_features = []

        feature = self.patches(x, selected_contrats)
        for layer in self.down_layers:
            skiped_features.append(feature)
            feature = layer(feature, selected_contrats)

        for layer in self.bottleneck:
            feature = layer(feature, selected_contrats)
        output_features.append(feature)

        for layer in self.up_layers:
            feature = layer(feature, skiped_features.pop(), selected_contrats)
            output_features.append(feature)

        return output_features


class DecoderUpLayer(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrats, num_heads, patch_size=2, reduction=True):
        super(DecoderUpLayer, self).__init__()
        self.layer = nn.Sequential(
            MultiContrastDecoderBlock(dim,
                                      window_size, (0, 0),
                                      num_contrats, num_heads),
            MultiContrastDecoderBlock(dim,
                                      window_size, shift_size,
                                      num_contrats, num_heads),
        )

        self.expands = PatchExpansion(dim, patch_size, reduction)

    def forward(self, x, skiped_features, selected_contrats):
        for layer in self.layer:
            x = layer(x, skiped_features, selected_contrats)
        x = self.expands(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_layers, window_size, shift_size, num_contrats, num_heads, patch_size=2):
        super(Decoder, self).__init__()

        self.up_layers = nn.ModuleList([
            DecoderUpLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                           num_contrats, num_heads, patch_size)
            for i in range(num_layers, 0, -1)
        ])

        self.expands = DecoderUpLayer(dim * patch_size ** 4, window_size, (0, 0),
                                      num_contrats, num_heads, patch_size * 2, reduction=False)

    def forward(self, x, encoded_features, selected_contrats):
        for layer in self.up_layers:
            x = layer(x, encoded_features.pop(0), selected_contrats)
        x = self.expands(x, encoded_features.pop(0), selected_contrats)

        return x
