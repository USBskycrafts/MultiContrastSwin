from .block import *
import torchvision


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
    """基于torchvision AlexNet的多尺度判别器（支持n_layers配置和ndf适配）"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, pretrained=True):
        super().__init__()
        self.n_layers = min(max(n_layers, 1), 5)  # 限制1-5层

        # 输入通道适配层
        self.adapt_conv = spectral_norm(
            nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0)
        )

        # 加载torchvision的AlexNet
        alexnet = torchvision.models.alexnet(pretrained=pretrained)
        alexnet_features = list(alexnet.features.children())

        # 构建判别器模型(只取前n_layers*2-1个层，因为每层包含卷积和激活)
        self.feature_layers = nn.ModuleList()
        for i in range(min(self.n_layers*3-1, len(alexnet_features))):
            layer = alexnet_features[i]
            self.feature_layers.append(layer)

        # 根据n_layers确定输出层输入通道数
        dim_map = {1: 64, 2: 192, 3: 384, 4: 256, 5: 256}
        output_dim = dim_map[self.n_layers]

        # PatchGAN输出层
        self.output_conv = spectral_norm(
            nn.Conv2d(output_dim, 1, kernel_size=4, stride=1, padding=2)
        )

    def forward(self, x):
        x = self.adapt_conv(x)
        for layer in self.feature_layers:
            x = layer(x)
        x = self.output_conv(x)
        return x
