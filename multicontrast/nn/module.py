import math
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


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class PixelDiscriminator(nn.Module):
    """pix2pixHD标准70x70 PatchGAN判别器(多尺度适配版)"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1

        # 输入适配层
        self.input_adapt = nn.Conv2d(
            input_nc, ndf, kernel_size=1, stride=1, padding=0)

        # 主网络结构
        sequence = [
            nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        # 中间卷积层
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 最后两层保持分辨率
        nf_mult = min(2 ** (n_layers-1), 8)  # 保持与前一层相同通道数
        sequence += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.input_adapt(x)
        return self.model(x)
