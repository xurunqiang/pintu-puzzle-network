"""
局部十字融合模块 (LocalFusion)
包含掩码深度可分离卷积与单层局部注意力 (1-layer Local Transformer Block)
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTransformerBlock(nn.Module):
    """
    单层 Pre-LN Transformer 模块:
    dim: 96, heads: 4, mlp_ratio: 2
    """
    def __init__(self, dim: int = 96, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, seq_len, dim)
        key_padding_mask: (B, seq_len), True 表示该位置为填充无效
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LocalFusion(nn.Module):
    """
    在 3x3 十字窗口 (目标块 + 上下左右有效邻居) 内执行局部融合:
    1. 掩码深度可分离卷积 (Depthwise + Pointwise Conv)
    2. 局部十字注意力 (1 层 Local Transformer Block)
    输出:
        更新后的目标块特征 (B, 96)
    """
    def __init__(self, dim: int = 96, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.dim = dim

        # 3x3 掩码深度可分离卷积
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(12, dim)
        self.act = nn.GELU()

        # 局部十字 Transformer
        self.local_transformer = LocalTransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        grid_features: torch.Tensor, # (B, dim, g, g) 当前整网格特征图
        grid_mask: torch.Tensor,     # (B, 1, g, g) 1 为已填入有效块，0 为尚未填入
        target_r: int,
        target_c: int,
        return_all: bool = False,
        valid_coords: Optional[set] = None
    ) -> torch.Tensor:
        """
        输入当前整图网格与新放置的目标槽位 (target_r, target_c)。
        仅提取其 3x3 局部十字窗口进行卷积与注意力融合，更新目标槽位特征。
        """
        B, C, g, _ = grid_features.shape

        # 1. 提取 3x3 局部窗口 (处理边界 padding: 用 0 填充边界外位置)
        # 裁剪范围: [target_r - 1 : target_r + 2, target_c - 1 : target_c + 2]
        r_start = max(0, target_r - 1)
        r_end = min(g, target_r + 2)
        c_start = max(0, target_c - 1)
        c_end = min(g, target_c + 2)

        # 构造局部 3x3 窗口
        local_feat = torch.zeros((B, C, 3, 3), device=grid_features.device, dtype=grid_features.dtype)
        local_mask = torch.zeros((B, 1, 3, 3), device=grid_features.device, dtype=grid_mask.dtype)

        # 将对应区域复制到 3x3 矩阵中
        dst_r_start = 1 - (target_r - r_start)
        dst_r_end = dst_r_start + (r_end - r_start)
        dst_c_start = 1 - (target_c - c_start)
        dst_c_end = dst_c_start + (c_end - c_start)

        local_feat[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid_features[:, :, r_start:r_end, c_start:c_end]
        local_mask[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid_mask[:, :, r_start:r_end, c_start:c_end]

        # 2. 十字掩码: 仅保留十字位置 (center, top, right, bottom, left)，忽略四角对角线
        cross_pattern = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]
        ], device=grid_features.device).view(1, 1, 3, 3)

        effective_mask = local_mask * cross_pattern # (B, 1, 3, 3)

        # 3. 掩码深度可分离卷积
        # 遮盖无效位置
        masked_input = local_feat * effective_mask
        counts = F.conv2d(effective_mask, torch.ones(1, 1, 3, 3,
                         device=grid_features.device, dtype=grid_features.dtype), padding=1)
        spatial = self.dw_conv(masked_input) * (9.0 / counts.clamp_min(1))
        conv_out = self.act(self.gn(self.pw_conv(spatial)))
        local_feat = local_feat + conv_out * effective_mask

        # 4. 局部十字注意力: 将十字内的 5 个候选位置作为序列进行自注意力
        # 十字相对坐标: center (1, 1), top (0, 1), right (1, 2), bottom (2, 1), left (1, 0)
        cross_coords = [(1, 1), (0, 1), (1, 2), (2, 1), (1, 0)]
        tokens = []
        token_masks = []
        for r_rel, c_rel in cross_coords:
            tok = local_feat[:, :, r_rel, c_rel] # (B, C)
            tok_m = effective_mask[:, 0, r_rel, c_rel] # (B,) 1 为有效，0 为无效
            tokens.append(tok)
            token_masks.append(tok_m)

        seq_tokens = torch.stack(tokens, dim=1) # (B, 5, C)
        # key_padding_mask: True 为无效
        pad_mask = torch.stack(token_masks, dim=1) < 0.5 # (B, 5)

        # 执行 1 层 Local Transformer
        trans_out = self.local_transformer(seq_tokens, key_padding_mask=pad_mask) # (B, 5, C)

        # 取出中心位置 (index 0) 的输出作为目标槽位融合后特征
        updated = self.out_norm(trans_out)
        if return_all:
            return {(target_r + r - 1, target_c + c - 1): updated[:, index]
                    for index, (r, c) in enumerate(cross_coords)
                    if 0 <= target_r + r - 1 < g and 0 <= target_c + c - 1 < g
                    and (((target_r + r - 1, target_c + c - 1) in valid_coords)
                         if valid_coords is not None else bool(effective_mask[:, 0, r, c].any()))}
        return updated[:, 0]
