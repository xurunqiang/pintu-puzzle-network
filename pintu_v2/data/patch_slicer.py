"""
数据切分、反射填充、有效像素掩码与 2 像素边带提取
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


class PatchSlicer:
    """
    针对 CIFAR-10 (32x32) 的非对称反射填充与切片工具。
    兼容 3x3, 5x5, 7x7 网格，并提取 2 像素宽的法向边带特征。
    """
    def __init__(self, grid_size: int = 5, edge_band_width: int = 2):
        self.grid_size = grid_size
        self.edge_band_width = edge_band_width
        
        # 对应规则配置
        if grid_size == 3:
            self.padded_size = (33, 33)
            self.patch_size = 11
            self.empty_count = 8
            self.pad = (0, 1, 0, 1) # (left, right, top, bottom)
        elif grid_size == 5:
            self.padded_size = (35, 35)
            self.patch_size = 7
            self.empty_count = 24
            self.pad = (1, 2, 1, 2)
        elif grid_size == 7:
            self.padded_size = (35, 35)
            self.patch_size = 5
            self.empty_count = 48
            self.pad = (1, 2, 1, 2)
        else:
            raise ValueError(f"不支持的网格大小: {grid_size}，必须为 3, 5, 或 7")
            
        self.center_coord = (grid_size // 2, grid_size // 2)

    def slice_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: img (B, 3, 32, 32)
        输出:
            patches: (B, grid_size, grid_size, 3, P, P)
            patch_masks: (B, grid_size, grid_size, 1, P, P) - 1 为真实像素，0 为人工反射填充
        """
        B, C, H, W = img.shape
        assert H == 32 and W == 32, f"输入尺寸必须为 32x32，当前为 {H}x{W}"

        # 1. 图像反射填充
        padded_img = F.pad(img, self.pad, mode='reflect')

        # 2. 构造有效像素掩码 (原图区域为 1，填充区域为 0)
        ones_mask = torch.ones((B, 1, 32, 32), device=img.device, dtype=img.dtype)
        padded_mask = F.pad(ones_mask, self.pad, mode='constant', value=0.0)

        # 3. 切割网格 (B, C, g, g, P, P) -> (B, g, g, C, P, P)
        P = self.patch_size
        patches = padded_img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5)
        patch_masks = padded_mask.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5)

        return patches, patch_masks

    def extract_edge_bands(self, patch: torch.Tensor, patch_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        提取单块的 4 个 2 像素宽边带，包含颜色与沿边界法向的梯度（共 6 通道）。
        输出全部统一变换为规范形状 (N, 6, 2, L)，其中 0 维为外边界像素，1 维为内部像素。
        
        参数:
            patch: (N, 3, P, P)
            patch_mask: (N, 1, P, P)，可选
        返回:
            dict of bands: {'top', 'right', 'bottom', 'left'}，每个张量形状为 (N, 6, 2, P)
            如果提供了 patch_mask，还返回 'masks': {'top', 'right', 'bottom', 'left'} (N, 1, 2, P)
        """
        N, C, H, W = patch.shape
        assert H == self.patch_size and W == self.patch_size
        L = self.patch_size

        # 1. Top 边带 (row 0 为边界, row 1 为内部；法向朝内向增量: row1 - row0)
        top_color = patch[:, :, :2, :] # (N, 3, 2, L)
        top_grad = (patch[:, :, 1:2, :] - patch[:, :, 0:1, :]).repeat(1, 1, 2, 1) # (N, 3, 2, L)
        top_band = torch.cat([top_color, top_grad], dim=1) # (N, 6, 2, L)

        # 2. Bottom 边带 (row -1 为边界, row -2 为内部；法向朝内向增量: row(-2) - row(-1))
        bot_color = torch.flip(patch[:, :, -2:, :], dims=[2]) # (N, 3, 2, L)
        bot_grad = (patch[:, :, -2:-1, :] - patch[:, :, -1:, :]).repeat(1, 1, 2, 1)
        bot_band = torch.cat([bot_color, bot_grad], dim=1)

        # 3. Left 边带 (col 0 为边界, col 1 为内部；法向朝内向增量: col1 - col0)
        left_color = patch[:, :, :, :2].permute(0, 1, 3, 2) # (N, 3, 2, L)
        left_grad = (patch[:, :, :, 1:2] - patch[:, :, :, 0:1]).permute(0, 1, 3, 2).repeat(1, 1, 2, 1)
        left_band = torch.cat([left_color, left_grad], dim=1)

        # 4. Right 边带 (col -1 为边界, col -2 为内部；法向朝内向增量: col(-2) - col(-1))
        right_color = torch.flip(patch[:, :, :, -2:], dims=[3]).permute(0, 1, 3, 2)
        right_grad = (patch[:, :, :, -2:-1] - patch[:, :, :, -1:]).permute(0, 1, 3, 2).repeat(1, 1, 2, 1)
        right_band = torch.cat([right_color, right_grad], dim=1)

        bands = {'top': top_band, 'right': right_band, 'bottom': bot_band, 'left': left_band}

        if patch_mask is not None:
            top_m = patch_mask[:, :, :2, :]
            bot_m = torch.flip(patch_mask[:, :, -2:, :], dims=[2])
            left_m = patch_mask[:, :, :, :2].permute(0, 1, 3, 2)
            right_m = torch.flip(patch_mask[:, :, :, -2:], dims=[3]).permute(0, 1, 3, 2)
            masks = {'top': top_m, 'right': right_m, 'bottom': bot_m, 'left': left_m}
            # 法向差分依赖两排像素；任一排无效时均不能作为真实梯度。
            for name, band in bands.items():
                pair_valid = masks[name][:, :, :1, :] * masks[name][:, :, 1:2, :]
                bands[name] = torch.cat((band[:, :3], band[:, 3:] * pair_valid), dim=1)
            return bands, masks

        return bands

    @staticmethod
    def compute_raw_seam_difference(band_a: torch.Tensor, band_b: torch.Tensor, 
                                     mask_a: torch.Tensor = None, mask_b: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算两个相邻边带之间的原始像素颜色误差与法向梯度误差。
        例如块 A 的右边带与块 B 的左边带接缝。
        band_a, band_b: (N, 6, 2, L)
        输出:
            color_diff: (N,) 颜色 L1 差异
            grad_diff: (N,) 梯度延续度差异
        """
        # 外边界颜色 (通道 0:3 的 0 维)
        color_a = band_a[:, :3, 0, :] # (N, 3, L)
        color_b = band_b[:, :3, 0, :]
        
        # 边界法向梯度 (通道 3:6 的 0 维)
        # 注意: 沿统一法线看，A 朝内与 B 朝内方向相反，接缝处平滑延续意味着 grad_a 应与 -grad_b 匹配
        grad_a = band_a[:, 3:6, 0, :]
        grad_b = band_b[:, 3:6, 0, :]

        # 有效像素权重
        if mask_a is not None and mask_b is not None:
            valid = (mask_a[:, 0, 0, :] * mask_b[:, 0, 0, :]).unsqueeze(1) # (N, 1, L)
            valid_count = (3 * valid.sum(dim=(1, 2))).clamp_min(1.0)
            c_diff = (torch.abs(color_a - color_b) * valid).sum(dim=(1, 2)) / valid_count
            grad_valid = valid * mask_a[:, :, 1, :] * mask_b[:, :, 1, :]
            grad_count = (3 * grad_valid.sum(dim=(1, 2))).clamp_min(1.0)
            g_diff = (torch.abs(grad_a + grad_b) * grad_valid).sum(dim=(1, 2)) / grad_count
        else:
            c_diff = torch.mean(torch.abs(color_a - color_b), dim=(1, 2))
            g_diff = torch.mean(torch.abs(grad_a + grad_b), dim=(1, 2))

        return c_diff, g_diff
