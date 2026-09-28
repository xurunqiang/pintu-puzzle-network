"""
内容卷积编码器 (ContentEncoder) 与辅助预测头 (位置预测、相对方向预测)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """
    内容卷积编码器:
    输入: 单个 Patch (B, 3, P, P)
    主干:
        Conv2d(3, 48, kernel_size=3, padding=1) -> GroupNorm(6, 48) -> GELU
        Conv2d(48, 96, kernel_size=3, padding=1) -> GroupNorm(12, 96) -> GELU
        AdaptiveAvgPool2d((2, 2)) -> Flatten (384)
        Linear(384, 96) -> 96 维内容特征
    辅助头:
        1. 归一化网格坐标预测头: Linear(96, 64) -> GELU -> Linear(64, 2) -> Sigmoid
        2. 相对方向分类头: Linear(192, 64) -> GELU -> Linear(64, 5) (上下左右/非相邻)
    """
    def __init__(self, content_dim: int = 96):
        super().__init__()
        self.content_dim = content_dim

        # 卷积主干
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(6, 48)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(48, content_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(12, content_dim)
        self.act2 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(content_dim * 2 * 2, content_dim)
        self.norm = nn.LayerNorm(content_dim)

        # 辅助头 1: 位置预测头 (输出 [0, 1]^2 归一化网格坐标)
        self.pos_head = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        # 辅助头 2: 相对方向预测头 (输入拼合两块的 96+96 维特征，预测 5 类相对关系)
        self.dir_head = nn.Sequential(
            nn.Linear(content_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播:
        输入: x: (B, 3, P, P)
        输出:
            feat: (B, 96) 内容特征
            pred_pos: (B, 2) 预测的归一化网格坐标
        """
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.act2(self.gn2(self.conv2(h)))
        h = self.pool(h) # (B, 96, 2, 2)
        h_flat = h.view(h.size(0), -1)
        feat = self.norm(self.proj(h_flat)) # (B, 96)

        pred_pos = self.pos_head(feat) # (B, 2)
        return feat, pred_pos

    def predict_direction(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        预测块 B 相对于块 A 的空间关系:
        0: 上, 1: 右, 2: 下, 3: 左, 4: 非相邻
        """
        cat_feat = torch.cat([feat_a, feat_b], dim=-1) # (B, 192)
        return self.dir_head(cat_feat) # (B, 5)
