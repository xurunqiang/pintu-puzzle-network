"""
全局网格融合模块 (GlobalFusion)
包含网格深度可分离卷积、2D 位置编码、双层全局 Transformer 与全局池化
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalFusion(nn.Module):
    """
    全网格拼图完成后的全局融合网络:
    输入:
        已组装完成的特征网格 (B, dim, g, g)，其中 dim=96, g=5 (默认)
    网络流程:
        1. 网格深度可分离卷积 (Depthwise + Pointwise Conv + GN + GELU) + 残差
        2. 叠加 2D 可学习位置编码 (B, dim, g, g)
        3. 展平为序列 (B, g*g, dim)
        4. 2 层全局 Transformer 编码器 (Pre-LN, 4 头, MLP 扩展 2 倍)
        5. 全局平均池化 (GAP) + LayerNorm -> 输出 96 维全图语义向量
    """
    def __init__(
        self,
        dim: int = 96,
        grid_size: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        # 1. 网格深度可分离卷积
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(12, dim)
        self.act = nn.GELU()

        # 2. 2D 位置编码 (1, dim, g, g)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, grid_size, grid_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. 2 层全局 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出归一化
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        """
        输入: grid_features (B, 96, g, g)
        输出: z_global (B, 96) 语义向量
        """
        B, C, H, W = grid_features.shape
        assert H == self.grid_size and W == self.grid_size, f"输入网格大小需为 {self.grid_size}x{self.grid_size}"

        # 1. 深度可分离卷积扩展感受野
        conv_out = self.act(self.gn(self.pw_conv(self.dw_conv(grid_features))))
        h = grid_features + conv_out

        # 2. 叠加 2D 位置编码
        h = h + self.pos_embed

        # 3. 展平为序列 (B, C, H, W) -> (B, H*W, C)
        seq = h.flatten(2).permute(0, 2, 1) # (B, g*g, 96)

        # 4. 双层 Transformer 全局上下文注意融合
        seq_out = self.transformer(seq) # (B, g*g, 96)

        # 5. 全局平均池化
        global_repr = self.out_norm(seq_out.mean(dim=1)) # (B, 96)

        return global_repr
