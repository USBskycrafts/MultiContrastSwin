from .module import *


class MultiContrastSwinTransformer(nn.Module):

    def __init__(self, dim, num_layers, window_size, shift_size, num_contrasts, num_heads, patch_size=2):
        super(MultiContrastSwinTransformer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_contrasts = num_contrasts
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.image_encoding = MultiContrastImageEncoding(
            in_channels=1, out_channels=dim, num_contrasts=num_contrasts)
        self.image_decoding = MultiContrastImageDecoding(
            in_channels=dim, out_channels=1, num_contrasts=num_contrasts)

        self.encoder = Encoder(dim, num_layers, window_size,
                               shift_size, num_contrasts, num_heads, patch_size)
        self.decoder = Decoder(dim, num_layers, window_size,
                               shift_size, num_contrasts, num_heads, patch_size)

        self.contrasts_seed = nn.Parameter(
            torch.randn(1, num_contrasts, 1, 1, dim *
                        (1 << (num_layers - 1)) * patch_size ** 4)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, selected_contrats, sample_times=1):
        x = self.image_encoding(x, selected_contrats[0])
        encoded_features = self.encoder(x, selected_contrats[0])
        B, M, H, W, C = encoded_features[0].shape

        seeds = self.contrasts_seed[:, selected_contrats[1], :]
        seeds = seeds.expand(B, -1, H, W, -1)  # 减少内存复制
        decoded_features = self.decoder(
            seeds, encoded_features, selected_contrats)
        y = self.image_decoding(decoded_features, selected_contrats[1])
        return y


class MultiContrastDiscriminator(nn.Module):
    def __init__(self,
                 dim,
                 num_layers,
                 window_size,
                 shift_size,
                 num_contrasts,
                 num_heads,
                 patch_size=2):
        super(MultiContrastDiscriminator, self).__init__()
        self.image_encoding = MultiContrastImageEncoding(
            in_channels=1, out_channels=dim, num_contrasts=num_contrasts)
        self.patches = EncoderDownLayer(
            dim, window_size, shift_size, num_contrasts, num_heads, patch_size * 2, False)
        num_layers -= 1
        self.down_layers = nn.ModuleList([
            EncoderDownLayer(dim * (1 << i) * patch_size ** 4, window_size, shift_size,
                             num_contrasts, num_heads, patch_size)
            for i in range(num_layers)
        ])

        self.out_proj = nn.Linear(dim * (1 << num_layers) * patch_size ** 4, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, std=0.02)

    def forward(self, x, generated_contrats):
        x = self.image_encoding(x, generated_contrats)
        x = self.patches(x, generated_contrats)
        for layer in self.down_layers:
            x = layer(x, generated_contrats)
        return self.out_proj(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers)
            self.discriminators.append(netD)

    def forward(self, x):
        outputs = []
        for scale, netD in enumerate(self.discriminators):
            # 下采样输入图像（多尺度处理）
            if scale > 0:
                x = nn.functional.interpolate(
                    x, scale_factor=0.5, mode='bilinear', align_corners=False
                )
            outputs.append(netD(x))
        return outputs  # 返回各尺度判别结果
