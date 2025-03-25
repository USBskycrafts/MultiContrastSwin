from .block import *


class EncoderDownLayer(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrasts, num_heads, patch_size=2, reduction=True):
        super(EncoderDownLayer, self).__init__()
        self.patches = PatchPartition(dim, patch_size, reduction)
        if reduction:
            dim *= patch_size
        else:
            dim *= (patch_size ** 2)
        self.layer = nn.Sequential(
            MultiContrastEncoderBlock(dim,
                                      window_size, (0, 0),
                                      num_contrasts, num_heads),
            MultiContrastEncoderBlock(dim,
                                      window_size, shift_size,
                                      num_contrasts, num_heads)
        )

    def forward(self, x, selected_contrats):
        h = self.patches(x)
        for layer in self.layer:
            h = layer(h, selected_contrats)
        return h


class EncoderUpLayer(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrasts, num_heads, patch_size=2, reduction=True):
        super(EncoderUpLayer, self).__init__()
        self.expands = PatchExpansion(dim, patch_size, reduction)
        self.layer = nn.Sequential(
            MultiContrastEncoderBlock(dim // patch_size,
                                      window_size, (0, 0),
                                      num_contrasts, num_heads),
            MultiContrastEncoderBlock(dim // patch_size,
                                      window_size, shift_size,
                                      num_contrasts, num_heads),
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
    def __init__(self, dim, num_layers, window_size, shift_size, num_contrasts, num_heads, patch_size=2):
        super(Encoder, self).__init__()
        # self.patches = PatchPartition(dim, patch_size * 2, False)
        self.patches = EncoderDownLayer(
            dim, window_size, shift_size, num_contrasts, num_heads, patch_size * 2, False)
        num_layers -= 1
        self.down_layers = nn.ModuleList([
            EncoderDownLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                             num_contrasts, num_heads * (1 << (i + 1)), patch_size)
            for i in range(num_layers)
        ])

        self.bottleneck = nn.Sequential(
            MultiContrastEncoderBlock(dim * (1 << num_layers) * patch_size ** 4,
                                      window_size, (0, 0),
                                      num_contrasts, num_heads * (1 << num_layers)),
            MultiContrastEncoderBlock(dim * (1 << num_layers) * patch_size ** 4,
                                      window_size, shift_size,
                                      num_contrasts, num_heads * (1 << num_layers)),
        )

        self.up_layers = nn.ModuleList([
            EncoderUpLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                           num_contrasts, num_heads * (1 << (i + 1)), patch_size)
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
    def __init__(self, dim, window_size, shift_size, num_contrasts, num_heads, patch_size=2, reduction=True):
        super(DecoderUpLayer, self).__init__()
        self.layer = nn.Sequential(
            MultiContrastDecoderBlock(dim,
                                      window_size, (0, 0),
                                      num_contrasts, num_heads),
            MultiContrastDecoderBlock(dim,
                                      window_size, shift_size,
                                      num_contrasts, num_heads),
        )

        self.expands = PatchExpansion(dim, patch_size, reduction)

    def forward(self, x, skiped_features, selected_contrats):
        for layer in self.layer:
            x = layer(x, skiped_features, selected_contrats)
        x = self.expands(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_layers, window_size, shift_size, num_contrasts, num_heads, patch_size=2):
        super(Decoder, self).__init__()
        num_layers -= 1
        self.up_layers = nn.ModuleList([
            DecoderUpLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                           num_contrasts, num_heads * (1 << (i + 1)), patch_size)
            for i in range(num_layers, 0, -1)
        ])

        self.expands = DecoderUpLayer(dim * patch_size ** 4, window_size, (0, 0),
                                      num_contrasts, num_heads, patch_size * 2, reduction=False)

    def forward(self, x, encoded_features, selected_contrats):
        for i, layer in enumerate(self.up_layers):
            x = layer(x, encoded_features[i], selected_contrats)
        x = self.expands(x, encoded_features[-1], selected_contrats)

        return x


class NLayerDiscriminator(nn.Module):
    """单尺度判别器（PatchGAN + 谱归一化）"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            # 第一层不使用谱归一化（原始论文设计）
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]

        # 中间层：逐步增加通道数
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                SpectralNormConv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=4, stride=2, padding=2
                )
            ]

        # 最后一层：输出1通道的Patch判别结果
        nf_mult_prev = nf_mult
        layers += [
            spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, 1,
                          kernel_size=4, stride=1, padding=2)
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
