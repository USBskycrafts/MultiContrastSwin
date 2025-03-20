from re import A
from typing import List

import torch
import torch.nn as nn

from .utils import *


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: Tuple[int, int], num_contrasts, num_heads, num_resouces):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_contrasts = num_contrasts
        self.num_resources = num_resouces

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.register_buffer('relative_bias_index',
                             get_relative_coords(num_contrasts, *window_size))

        table_size = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * num_contrasts - 1)
        self.relative_bias_table = nn.Parameter(
            torch.randn(table_size, num_heads))

        self.embeddings = nn.Parameter(torch.randn(
            num_resouces, 2 ** num_contrasts, 1, 1, dim))

    def forward(self, q, k, v, selected_contrasts: Tuple[List[int], List[int]], mask=None):
        assert k.shape == v.shape
        assert q.shape[1] == len(
            selected_contrasts[0]), f'{q.shape[1]} != {len(selected_contrasts[0])}'
        assert k.shape[1] == len(
            selected_contrasts[1]), f'{k.shape[1]} != {len(selected_contrasts[1])}'

        M, N = q.shape[1], k.shape[1]
        H, W = q.shape[2], q.shape[3]
        q, k, v = map(lambda x: window_partition(
            x, self.window_size), (q, k, v))

        q = q + select_embeddings(self.embeddings[0], selected_contrasts[0])
        k = k + select_embeddings(self.embeddings[1] if self.num_resources > 1 else self.embeddings[0],
                                  selected_contrasts[1])
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        relative_bias_index = select_relative_coords(
            self.relative_bias_index,
            self.num_contrasts, self.window_size[0], self.window_size[1], selected_contrasts
        )
        relative_bias_index = relative_bias_index.view(-1)
        relative_bias = self.relative_bias_table[relative_bias_index].view(
            M * self.window_size[0] * self.window_size[1],
            N * self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1)

        q, k, v = map(lambda x: multihead_shuffle(
            x, self.num_heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-2, -1)) * \
            self.scale + relative_bias
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        x = multihead_unshuffle(x, self.num_heads)
        x = window_reverse(x, self.window_size, M, H, W)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.mlp(x)


class MoELayer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, use_aux_loss=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        # self.k = k
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [MLP(input_size, 2 * input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)

        if not use_aux_loss:
            self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def _update_expert_biases(self, update_rate=1e-3):
        expert_counts = torch.bincount(self.top_k_indices.flatten(),
                                       minlength=self.num_experts)
        avg_count = expert_counts.float().mean()
        for i, count in enumerate(expert_counts):
            # b_i = b_i + u + sign(e_i)
            # note: this is \bar{c_i} - c_i, NOT c_i - \bar{c_i}, which will push the network to
            # be maximally unbalanced. Really important to get this part right!!!
            error = avg_count - count.float()
            self.expert_biases.data[i] += update_rate * torch.sign(
                error)

    def forward(self, x):
        # s_{i,t}
        B, M, H, W, C = x.shape
        gate_output = self.gate(x)
        # use sigmoid gate instead of softmax
        gate_probs = torch.sigmoid(gate_output)

        # do top k based on s_{i,t} + b_i
        if not self.use_aux_loss:
            gate_logits = gate_output + self.expert_biases
        else:
            gate_logits = gate_output

        _, top_k_indices = torch.topk(gate_logits, M, dim=-1)

        # ...but make sure we use the unbiased s_{i,t} as the gate value
        top_k_probs = gate_probs.gather(-1, top_k_indices)

        # normalize to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # get the routed expert outputs
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts])
        indices = top_k_indices.unsqueeze(-1).expand(-1, -
                                                     1, -1, -1, -1, self.output_size)
        expert_outputs = expert_outputs.gather(
            0, indices.permute(5, 0, 1, 2, 3, 4)).permute(1, 2, 3, 4, 5, 0)

        final_output = (expert_outputs *
                        top_k_probs.unsqueeze(-1)).sum(dim=-2)
        self.top_k_indices = top_k_indices
        if self.training:
            self._update_expert_biases()
        return final_output


class MultiContrastEncoderBlock(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrasts, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_contrasts = num_contrasts
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_contrasts, num_heads, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MoELayer(dim, dim, num_contrasts)

    def forward(self, x, selected_contrasts):
        B, M, H, W, C = x.shape
        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))

        selected_contrasts = (selected_contrasts, selected_contrasts)
        mask = create_attention_mask(
            M, M, H, W, self.num_heads, self.window_size, self.shift_size)
        mask = mask.to(x.device)
        x = x + self.attn(self.norm1(x), self.norm1(x),
                          self.norm1(x), selected_contrasts, mask)

        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))

        x = x + self.mlp(self.norm2(x))
        return x


class MultiContrastDecoderBlock(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_contrasts, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_contrasts = num_contrasts
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = WindowAttention(
            dim, window_size, num_contrasts, num_heads, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = WindowAttention(
            dim, window_size, num_contrasts, num_heads, 2)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, encoded_features, selected_contrasts):
        B, M, H, W, C = x.shape
        N = encoded_features.shape[1]
        assert x.shape[2:] == encoded_features.shape[2:], print(
            x.shape, encoded_features.shape)

        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        mask = create_attention_mask(
            M, M, H, W, self.num_heads, self.window_size, self.shift_size)
        mask = mask.to(x.device)
        x = x + self.attn1(self.norm1(x), self.norm1(x),
                           self.norm1(x), (selected_contrasts[1], selected_contrasts[1]), mask)

        mask = create_attention_mask(
            M, N, H, W, self.num_heads, self.window_size, self.shift_size)
        mask = mask.to(x.device)
        x = x + self.attn2(self.norm2(x), encoded_features,
                           encoded_features, (selected_contrasts[1], selected_contrasts[0]), mask)

        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))

        x = x + self.mlp(self.norm3(x))
        return x


class PatchPartition(nn.Module):
    def __init__(self, dim, patch_size, reduction=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.reduction = reduction

        if reduction:
            self.proj = nn.Sequential(
                nn.LayerNorm(dim * (patch_size ** 2)),
                nn.Linear(dim * (patch_size ** 2), dim * patch_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(dim * (patch_size ** 2)),
                nn.Linear(dim * (patch_size ** 2), dim * (patch_size ** 2)),
            )

    def forward(self, x):
        B, M, H, W, C = x.shape
        x = x.reshape(B, M, H // self.patch_size, self.patch_size,
                      W // self.patch_size, self.patch_size, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, M, H // self.patch_size, W // self.patch_size, -1)
        x = self.proj(x)
        return x


class PatchExpansion(nn.Module):
    def __init__(self, dim, patch_size, reduction=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.reduction = reduction

        if reduction:
            self.proj = nn.Sequential(
                nn.LayerNorm(dim // (patch_size ** 2)),
                nn.Linear(dim // (patch_size ** 2), dim // patch_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(dim // (patch_size ** 2)),
                nn.Linear(dim // (patch_size ** 2), dim // (patch_size ** 2)),
            )

    def forward(self, x):
        B, M, H, W, C = x.shape
        x = x.view(B, M, H, W, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, M, H * self.patch_size, W * self.patch_size, -1)
        x = self.proj(x)
        return x


class ImageEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.inc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, 7),
            # nn.PReLU(out_channels)
            nn.LeakyReLU(inplace=True)
        )

        self.convs = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(out_channels, out_channels, 5),
            # nn.PReLU(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(out_channels, out_channels, 5),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.inc(x)
        x = x + self.convs(x)
        return x.permute(0, 2, 3, 1).contiguous()


class ImageDecoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, in_channels, 5),
            # nn.PReLU(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, in_channels, 5),
        )

        self.outc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.convs(x) + x
        return self.outc(x).permute(0, 2, 3, 1).contiguous()


class MultiContrastImageEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, num_contrasts):
        super().__init__()

        self.encodings = nn.ModuleList(
            [ImageEncoding(in_channels, out_channels)
             for _ in range(num_contrasts)]
        )

    def forward(self, x, selected_contrats):
        return torch.stack([self.encodings[offset](x[:, i, ...]) for i, offset in enumerate(selected_contrats)], dim=1)


class MultiContrastImageDecoding(nn.Module):
    def __init__(self, in_channels, out_channels, num_contrasts):
        super().__init__()

        self.decodings = nn.ModuleList(
            [ImageDecoding(in_channels, out_channels)
             for _ in range(num_contrasts)]
        )

    def forward(self, x, selected_contrats: List[int]):
        return torch.stack([self.decodings[offset](x[:, i, ...]) for i, offset in enumerate(selected_contrats)], dim=1)
