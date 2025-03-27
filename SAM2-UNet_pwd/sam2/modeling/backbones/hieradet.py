# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


# ... (기존 import 및 do_pool 함수는 그대로 유지) ...

class MultiScaleBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, norm_layer: Union[nn.Module, str] = "LayerNorm",
                 q_stride: Tuple[int, int] = None, act_layer: nn.Module = nn.GELU,
                 window_size: int = 0):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Input x shape: {x.shape}")  # 입력 차원 출력
        shortcut = x  # B, H, W, C
        x = self.norm1(x)
        # # print(f"After norm1 x shape: {x.shape}")  # norm1 후 차원 출력

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
            # print(f"Shortcut after proj and pool: {shortcut.shape}")  # proj와 pool 후 차원 출력

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
            # print(f"After window_partition x shape: {x.shape}")  # window_partition 후 차원 출력

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        # print(f"After attn x shape: {x.shape}")  # attn 후 차원 출력
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
            # print(f"After attn with q_pool: {x.shape}, H={H}, W={W}")  # q_pool 적용 시 차원 출력

        # Reverse window partition
        if self.window_size > 0:
            # print(f"Before window_unpartition x shape: {x.shape}")  # window_unpartition 전 차원 출력
            x = window_unpartition(x, window_size, pad_hw, (H, W))
            # print(f"After window_unpartition x shape: {x.shape}")  # window_unpartition 후 차원 출력

        x = shortcut + self.drop_path(x)
        # print(f"After shortcut addition x shape: {x.shape}")  # shortcut 더한 후 차원 출력
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print(f"Output x shape: {x.shape}")  # 최종 출력 차원 출력
        return x


class Hiera(nn.Module):
    def __init__(self, embed_dim: int = 96, num_heads: int = 1, drop_path_rate: float = 0.0,
                 q_pool: int = 3, q_stride: Tuple[int, int] = (2, 2), stages: Tuple[int, ...] = (2, 3, 16, 3),
                 dim_mul: float = 2.0, head_mul: float = 2.0, window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
                 window_spec: Tuple[int, ...] = (8, 4, 14, 7), global_att_blocks: Tuple[int, ...] = (12, 16, 20),
                 return_interm_layers=True):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if i in self.global_att_blocks:
                window_size = 0
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads,
                drop_path=dpr[i], q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]] if return_interm_layers else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # print(f"After patch_embed: {x.shape}")
        x = x + self._get_pos_embed(x.shape[1:3])
        # print(f"After pos_embed: {x.shape}")

        outputs = []
        cur_stage = 0
        for i, blk in enumerate(self.blocks):
            # print(f"\nBlock {i} (Stage {cur_stage}) start: {x.shape}")
            x = blk(x)
            # print(f"Block {i} (Stage {cur_stage}) end: {x.shape}")
            if i in self.stage_ends:
                cur_stage += 1
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                # print(f"Output feats at block {i} (Stage {cur_stage}): {feats.shape}")
                outputs.append(feats)

        return outputs


# ... (나머지 코드는 그대로) ...