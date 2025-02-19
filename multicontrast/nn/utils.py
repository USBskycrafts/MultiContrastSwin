from typing import List, Tuple


import torch


def window_partition(x, window_size):
    B, M, H, W, C = x.shape
    x = x.view(B, M, H // window_size[0], window_size[0],
               W // window_size[1], window_size[1], C)
    x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    return x.view(-1, M * window_size[0] * window_size[1], C)


def window_reverse(windows, window_size, M, H, W):
    B = windows.shape[0] // (H * W // window_size[0] // window_size[1])
    x = windows.view(
        B, H // window_size[0], W // window_size[1], M, window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    return x.view(B, M, H, W, -1)


def get_relative_coords(M, H, W):
    coords_contrast = torch.arange(M)
    coords_h = torch.arange(H)
    coords_w = torch.arange(W)
    coords = torch.stack(torch.meshgrid([coords_contrast, coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    # [3, M * H * W, M * H * W]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += M - 1
    relative_coords[:, :, 1] += H - 1
    relative_coords[:, :, 2] += W - 1

    relative_coords[:, :, 0] *= (2 * H - 1) * (2 * W - 1)
    relative_coords[:, :, 1] *= (2 * W - 1)
    relative_coords = relative_coords.sum(-1)
    # [M * H * W, M * H * W]
    return relative_coords


def select_relative_coords(relative_coords, num_contrasts, H, W, selected_indices: Tuple[List[int], List[int]]):
    relative_coords = relative_coords.view(
        num_contrasts, H * W, num_contrasts, H * W)
    relative_coords = relative_coords[selected_indices[0], ...]
    relative_coords = relative_coords[:, :, selected_indices[1], :]
    return relative_coords.view(len(selected_indices[0]), H * W, len(selected_indices[1]), H * W)


def select_embeddings(embeddings, selected_indices: List[int]):
    """
    embeddings is a [B, 1 << M, 1, 1, C] tensor
    offest is a [1 << contrast_nun, dim] tensor

    """
    offset = sum(1 << index for index in selected_indices)
    return embeddings[offset, ...]


def multihead_shuffle(x, num_heads):
    B, N, C = x.shape
    x = x.view(B, N, num_heads, C // num_heads)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def multihead_unshuffle(x, num_heads):
    B, *_, C = x.shape
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(B, -1, C * num_heads)


def create_attention_mask(M, N, H, W, num_heads, window_size: Tuple[int, int], shift_size: Tuple[int, int]):
    img_mask = torch.zeros((1, 1, H, W, 1))

    h_slices = (slice(0, -window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    w_slices = (slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w, :] = cnt
            cnt += 1

    q_proj = img_mask.repeat(1, M, 1, 1, 1)
    kv_proj = img_mask.repeat(1, N, 1, 1, 1)

    q_proj = window_partition(q_proj, window_size)
    kv_proj = window_partition(kv_proj, window_size)

    q_proj = q_proj.view(-1, 1,  M * window_size[0] * window_size[1])
    kv_proj = kv_proj.view(-1, 1, N * window_size[0] * window_size[1])
    # or attn_mask = q_proj[:, :, :, None] - kv_proj[:, :, None, :] ?
    attn_mask = q_proj[0:1, :, :, None] - kv_proj[0:1, :, None, :]
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
