"""
“拼图”网络第二版：全局配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class SlicerConfig:
    # 支持的网格配置: grid_size -> (padded_h_w, patch_size, empty_count, pad_tuple)
    # pad_tuple: (left, right, top, bottom)
    GRID_SPECS: Dict[int, Dict] = field(default_factory=lambda: {
        3: {
            "padded_size": (33, 33),
            "patch_size": 11,
            "empty_count": 8,
            "pad": (0, 1, 0, 1), # 右侧、下侧补 1
        },
        5: {
            "padded_size": (35, 35),
            "patch_size": 7,
            "empty_count": 24,
            "pad": (1, 2, 1, 2), # 尽量对称，奇数差额放右侧、下侧
        },
        7: {
            "padded_size": (35, 35),
            "patch_size": 5,
            "empty_count": 48,
            "pad": (1, 2, 1, 2),
        }
    })
    edge_band_width: int = 2  # 边带宽 (像素)


@dataclass
class ModelConfig:
    grid_size: int = 5                  # 默认 5x5 网格
    content_dim: int = 96               # 96 维内容特征
    edge_dim: int = 32                  # 单个边缘 32 维特征
    local_heads: int = 4                # 局部 Transformer 头数
    local_mlp_ratio: int = 2            # 局部 Transformer MLP 扩展率
    global_layers: int = 2              # 全局 Transformer 层数
    global_heads: int = 4               # 全局 Transformer 头数
    global_mlp_ratio: int = 2           # 全局 Transformer MLP 扩展率
    pos_penalty_weight: float = 0.1     # 位置偏置惩罚权重
    tau_start: float = 1.0              # 软选择起始温度
    tau_end: float = 0.2                # 软选择最终温度
    edge_gradient_scale: float = 0.1    # 接缝梯度权重的响应尺度
    seam_color_weight: float = 0.25
    seam_gradient_weight: float = 0.15
    missing_edge_penalty: float = 0.5   # 内部连接方向缺少真实边缘证据的惩罚
    seed_anchor_weight: float = 0.2     # 防止上下文检索逐步偏离种子语义
    repetition_score_weight: float = 0.2 # 复用模式按硬使用次数降低候选分数
    direction_chunk_size: int = 512


@dataclass
class PoolConfig:
    pool_size: int = 8192               # 8,192 个唯一候选 Patch
    momentum_decay: float = 0.99        # 动量编码器更新系数
    feature_refresh_steps: int = 100    # 每 100 步刷新池中特征
    seed: int = 42                      # 固定随机种子建池
    consumed_inventory: bool = True     # 默认当前批次消耗模式
    allow_self_patches: bool = True     # 是否允许候选池挑选自身原图的 Patch（拼回原图）
    inventory_scope: str = "batch"      # batch: 批次共享消耗；image: 每张图独立消耗


@dataclass
class TrainConfig:
    data_dir: str = "./data/cifar10"
    save_dir: str = "./checkpoints/pintu_v2"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 20             # 前 20 轮预热基础编码器
    full_epochs: int = 80               # 后 80 轮完整拼图训练
    total_epochs: int = 100
    lr_warmup_epochs: int = 5           # 学习率 warmup 轮数
    num_workers: int = 4
    seed: int = 42

    # 损失权重
    # V = L_invar + L_var + 0.04 * L_cov
    lambda_invar: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.04

    # 总损失组合
    w_vicreg_full: float = 1.0
    w_vicreg_puzzle: float = 1.0
    w_direction: float = 1.0
    w_edge: float = 0.5
    w_position: float = 0.1
    w_align: float = 0.5
    w_rep_penalty: float = 0.05         # 不消耗模式下的重复惩罚
