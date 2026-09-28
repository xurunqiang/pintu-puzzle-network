"""
统一边缘卷积编码器 (EdgeEncoder)
前置掩码屏蔽、掩码传播、有效性检测与无效边带排除
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEncoder(nn.Module):
    """
    通用边缘卷积编码器:
    输入: 规范形状的 2 像素宽边带 (N, 6, 2, P)
          6 通道 = 3 通道颜色 + 3 通道法向差分梯度
          mask: (N, 1, 2, P) 有效像素掩码 (1 为真实像素，0 为人工反射填充)
    修复改进:
        1. 卷积前先将无效像素严格置 0 (输入端掩码隔离)；
        2. 卷积过程中逐层屏蔽无效感受野扩展；
        3. 仅对真实有效列进行特征加权池化；
        4. 全无效边带输出全 0 特征并标记 edge_valid_flag=False，防止作为有效单位向量参与误评分。
    """
    def __init__(self, edge_dim: int = 32):
        super().__init__()
        self.edge_dim = edge_dim

        self.conv1 = nn.Conv2d(6, 16, kernel_size=(2, 3), padding=(0, 1), bias=False)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(16, edge_dim, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.act2 = nn.GELU()

        self.proj = nn.Linear(edge_dim, edge_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算:
        输入:
            x: (N, 6, 2, P)
            mask: (N, 1, 2, P) 可选有效像素掩码
        输出:
            feat: (N, 32) 边缘特征 (有效边带为 L2 归一化单位向量，全无效边带为 0 向量)
            edge_valid_flag: (N,) bool 指示该边带是否拥有有效图像像素
        """
        N, _, H, W = x.shape
        device = x.device

        if mask is not None:
            # 1. 输入前置隔离: 屏蔽人工反射填充像素
            x = x * mask
            # 判定外边界 (row 0) 的有效性分布 (N, 1, 1, W)
            w_valid = (mask[:, :, 0:1, :] > 0.5).float() # (N, 1, 1, W)
            edge_valid_flag = (w_valid.sum(dim=(1, 2, 3)) > 0) # (N,)
        else:
            w_valid = torch.ones((N, 1, 1, W), device=device, dtype=torch.float32)
            edge_valid_flag = torch.ones((N,), device=device, dtype=torch.bool)

        # 2. 逐层卷积计算并在每层后抑制无效区域
        h = self.act1(self.conv1(x)) # (N, 16, 1, W)
        h = h * w_valid

        h = self.act2(self.conv2(h)) # (N, 32, 1, W)
        h = h * w_valid

        # 3. 掩码池化
        w_sum = torch.clamp(w_valid.sum(dim=-1, keepdim=True), min=1.0) # (N, 1, 1, 1)
        pooled = (h * w_valid).sum(dim=-1, keepdim=True) / w_sum        # (N, 32, 1, 1)
        pooled = pooled.view(N, self.edge_dim)

        feat = self.proj(pooled)

        # 4. 针对有效边做 L2 归一化；对完全无效边直接置 0 向量
        norm_feat = F.normalize(feat, p=2, dim=-1)
        norm_feat = norm_feat * edge_valid_flag.unsqueeze(1).float()

        return norm_feat, edge_valid_flag
