from torch import Tensor
from torch import nn
import functools
import operator
from re import A
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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

        self.scale = torch.rsqrt(torch.tensor(
            self.head_dim, dtype=torch.float32))

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
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)
        # x = F.scaled_dot_product_attention(
        #     q, k, v, attn_mask=relative_bias +
        #     (mask if mask is not None else 0),
        #     dropout_p=0.1 if self.attn_drop else 0.0,
        # )

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
    def __init__(self, input_size, output_size, num_contrasts, k=2, use_aux_loss=False):
        if not dist.is_initialized():
            raise RuntimeError(
                "MoELayer requires torch.distributed to be initialized first")
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        num_experts = 1 << num_contrasts - 2
        self.num_experts = num_experts
        self.k = k  # 明确指定选择的专家数量
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [MLP(input_size, 2 * input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)

        if not use_aux_loss:
            self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def _update_expert_biases(self, top_k_indices, update_rate=1e-3):
        # 跨进程聚合专家选择
        expert_counts = torch.bincount(
            top_k_indices.flatten(), minlength=self.num_experts).to(self.expert_biases.device)

        # 全局聚合统计
        dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)

        # 计算进程平均
        avg_count = expert_counts.float().mean() / dist.get_world_size()

        # 同步更新参数
        for i in range(self.num_experts):
            error = avg_count - expert_counts[i].float()
            self.expert_biases.data[i] += update_rate * torch.sign(error)

    def forward(self, x):
        original_shape = x.shape
        B, M, H, W, C = original_shape

        # 展平空间和位置维度
        x_flat = x.view(-1, C)  # (B*M*H*W, C)

        # 计算门控输出
        gate_output = self.gate(x_flat)  # (B*M*H*W, num_experts)

        # 应用专家偏置（如果启用）
        if not self.use_aux_loss:
            gate_logits = gate_output + self.expert_biases
        else:
            gate_logits = gate_output

        # 选择top-k专家
        top_k_values, top_k_indices = torch.topk(
            gate_logits, self.k, dim=-1)  # (B*M*H*W, k)

        # 计算门控概率并归一化
        gate_probs = torch.sigmoid(top_k_values)
        gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True)  # 归一化

        # 准备索引数据
        flat_indices = top_k_indices.view(-1)  # (B*M*H*W*k,)
        batch_indices = torch.arange(x_flat.size(
            0), device=x.device).repeat_interleave(self.k)  # (B*M*H*W*k,)
        probs_flat = gate_probs.view(-1)  # (B*M*H*W*k,)

        # 找出唯一需要计算的专家
        unique_experts = torch.unique(flat_indices)

        # 初始化输出
        final_output = torch.zeros_like(x_flat)  # (B*M*H*W, output_size)

        # 动态计算被选中的专家
        for expert_idx in unique_experts:
            # 找出当前专家需要处理的位置
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue

            # 获取对应的输入和权重
            expert_input = x_flat[batch_indices[mask]]
            expert_weight = probs_flat[mask].unsqueeze(-1)

            # 计算专家输出并加权
            expert_output = self.experts[expert_idx](expert_input)
            expanded_indices = batch_indices[mask].unsqueeze(
                -1).expand(-1, expert_output.size(-1))

            # Use scatter_add with proper dimension handling
            final_output.scatter_add_(
                0,
                expanded_indices,
                expert_output * expert_weight
            )
        # 恢复原始形状
        final_output = final_output.view(
            *original_shape[:-1], self.output_size)
        if self.training and not self.use_aux_loss:
            current_top_k = top_k_indices.view(original_shape[:-1] + (self.k,))
            self._update_expert_biases(current_top_k)

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
        self.mlp = MLP(dim, 4 * dim, dim)

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
        self.mlp = MLP(dim, 4 * dim, dim)

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
            nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.convs = nn.Sequential(
            nn.Conv2d(out_channels, 4 * out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
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
            nn.Conv2d(in_channels, in_channels * 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(4 * in_channels, in_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.outc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 3),
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
