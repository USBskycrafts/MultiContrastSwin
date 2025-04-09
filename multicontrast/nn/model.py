from networkx import reverse
from .module import *
import torch.nn.functional as F


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

        self.contrasts_seeds = nn.Parameter(
            torch.randn(1, num_contrasts, 1, 1, dim *
                        (1 << (num_layers - 1)) * patch_size ** 4)
        )

        self.quantizers = nn.ModuleList([
            VectorQuantizer(dim * (1 << i) * patch_size ** 4,
                            dim * (1 << (num_layers - i - 1)), 1e-2)
            for i in range(num_layers)
        ])

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

        seeds = self.contrasts_seeds[:, selected_contrats[1], :]
        seeds = seeds.expand(B, -1, H, W, -1)  # 减少内存复制
        quantized_features = []

        for encoded_feature, qt in zip(encoded_features, reversed(self.quantizers)):
            z_q = qt(encoded_feature)
            quantized_features.append(z_q)

        encoded_features = quantized_features
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
    def __init__(self, input_nc=1, ndf=64, n_layers=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        # 创建多尺度判别器（原始、1/2、1/4分辨率）
        for _ in range(num_scales):
            netD = PixelDiscriminator(input_nc, ndf, n_layers)
            self.discriminators.append(netD)

    def forward(self, x):
        outputs = []
        # 生成多尺度输入
        input_downsampled = x
        for netD in self.discriminators:
            outputs.append(netD(input_downsampled))
            # 下采样到下一个尺度
            input_downsampled = F.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear')
        return outputs
