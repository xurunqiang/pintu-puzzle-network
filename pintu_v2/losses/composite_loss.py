"""
多任务综合损失函数 (CompositeLoss)
实现解耦的拼图双视图自监督 VICReg、全图教师对齐、均衡方向分类、边缘对比与可微重复惩罚
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pintu_v2.config import TrainConfig
from pintu_v2.losses.vicreg import VICRegLoss


class CompositeLoss(nn.Module):
    """
    自监督综合训练损失:
    1. L_vicreg_full: 增强全图两视图 (z_v1, z_v2) 之间的 VICReg 损失
    2. L_vicreg_puzzle: 双拼图视图 (z_puzzle1, z_puzzle2) 之间的一致性 VICReg 损失 (解耦设计)
    3. L_align: 拼图表示与动量教师全图表示 (z_teacher) 之间的余弦对齐损失
    4. L_edge: 真实相邻 Patch 接缝边缘的对比学习损失 (InfoNCE)
    5. L_direction: 空间相对方位 5 分类 (上、右、下、左、非相邻) 交叉熵损失
    6. L_position: 块绝对网格坐标预测的 Smooth L1 损失
    7. L_rep: 在不消耗库存模式下的连续可微重复使用惩罚
    """
    def __init__(self, cfg: Optional[TrainConfig] = None):
        super().__init__()
        self.cfg = cfg or TrainConfig()

        self.vicreg = VICRegLoss(
            lambda_invar=self.cfg.lambda_invar,
            lambda_var=self.cfg.lambda_var,
            lambda_cov=self.cfg.lambda_cov
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def compute_edge_contrastive_loss(
        self,
        edge_feats_a: torch.Tensor, # (N, 32)
        edge_feats_b: torch.Tensor, # (N, 32) 正样本对 (真实物理接缝对应两侧)
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        InfoNCE 边带对比学习损失。
        """
        N = edge_feats_a.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=edge_feats_a.device)

        sim_matrix = torch.matmul(edge_feats_a, edge_feats_b.t()) / temperature
        labels = torch.arange(N, device=edge_feats_a.device)
        loss = self.ce_loss(sim_matrix, labels)
        return loss

    def forward(
        self,
        phase: str, # "warmup" 或 "full"
        z_v1: torch.Tensor,
        z_v2: torch.Tensor,
        edge_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dir_logits: Optional[torch.Tensor] = None,
        dir_targets: Optional[torch.Tensor] = None,
        pred_pos: Optional[torch.Tensor] = None,
        target_pos: Optional[torch.Tensor] = None,
        z_puzzle1: Optional[torch.Tensor] = None,
        z_puzzle2: Optional[torch.Tensor] = None,
        z_teacher: Optional[torch.Tensor] = None,
        repetition_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失。
        """
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=z_v1.device)

        # 1. 全图两视图基础 VICReg 损失
        l_vicreg_full, v_details = self.vicreg(z_v1, z_v2)
        total_loss = total_loss + self.cfg.w_vicreg_full * l_vicreg_full
        loss_dict["loss_vicreg_full"] = l_vicreg_full.item()
        loss_dict["invar_full"] = v_details["loss_invar"].item()
        loss_dict["var_full"] = v_details["loss_var"].item()
        loss_dict["cov_full"] = v_details["loss_cov"].item()

        # 2. 边缘对比损失
        if edge_pairs is not None:
            e_a, e_b = edge_pairs
            l_edge = self.compute_edge_contrastive_loss(e_a, e_b)
            total_loss = total_loss + self.cfg.w_edge * l_edge
            loss_dict["loss_edge"] = l_edge.item()

        # 3. 相对方向 5 分类损失
        if dir_logits is not None and dir_targets is not None:
            l_dir = self.ce_loss(dir_logits, dir_targets)
            total_loss = total_loss + self.cfg.w_direction * l_dir
            loss_dict["loss_dir"] = l_dir.item()

        # 4. 绝对网格坐标预测损失
        if pred_pos is not None and target_pos is not None:
            l_pos = self.smooth_l1(pred_pos, target_pos)
            total_loss = total_loss + self.cfg.w_position * l_pos
            loss_dict["loss_pos"] = l_pos.item()

        # 5. 第二阶段专属: 双拼图视图 VICReg 与教师对齐损失
        if phase == "full":
            # 5.1 双拼图一致性 VICReg 损失
            if z_puzzle1 is not None and z_puzzle2 is not None:
                l_vicreg_puz, p_details = self.vicreg(z_puzzle1, z_puzzle2)
                total_loss = total_loss + self.cfg.w_vicreg_puzzle * l_vicreg_puz
                loss_dict["loss_vicreg_puzzle"] = l_vicreg_puz.item()
                loss_dict["invar_puzzle"] = p_details["loss_invar"].item()
                loss_dict["var_puzzle"] = p_details["loss_var"].item()
            elif z_puzzle1 is not None and z_teacher is not None:
                # 兼容单拼图输入时与教师的正则
                l_vicreg_puz, _ = self.vicreg(z_puzzle1, z_teacher)
                total_loss = total_loss + self.cfg.w_vicreg_puzzle * l_vicreg_puz
                loss_dict["loss_vicreg_puzzle"] = l_vicreg_puz.item()

            # 5.2 拼图与动量全图教师的余弦对齐损失
            if z_puzzle1 is not None and z_teacher is not None:
                cos_sim = F.cosine_similarity(z_puzzle1, z_teacher, dim=-1).mean()
                l_align = 1.0 - cos_sim
                total_loss = total_loss + self.cfg.w_align * l_align
                loss_dict["loss_align"] = l_align.item()

            # 5.3 连续可微重复使用惩罚 (不消耗模式)
            if repetition_loss is not None and self.cfg.w_rep_penalty > 0:
                total_loss = total_loss + self.cfg.w_rep_penalty * repetition_loss
                loss_dict["loss_rep"] = repetition_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
