"""
VICReg 自监督损失函数 (Variance-Invariance-Covariance Regularization)
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg 损失计算模块:
    1. 不变性损失 (Invariance): 两视角表示之间的均方误差 MSE
    2. 方差损失 (Variance): 沿批次维度的标准差 hinge 惩罚，防止坍塌为单点
    3. 协方差损失 (Covariance): 特征维度间的协方差非对角线惩罚，去冗余与去相关
    """
    def __init__(
        self,
        lambda_invar: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        gamma: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.lambda_invar = lambda_invar
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        输入: z1, z2 均为 (B, D) 表示向量
        输出:
            total_loss: 标量总损失
            loss_dict: 各分项明细
        """
        B, D = z1.shape
        assert B > 1, f"VICReg 需 batch_size > 1 以计算方差和协方差，当前为 {B}"

        # 1. 不变性损失 L_invar
        sim_loss = F.mse_loss(z1, z2)

        # 2. 方差损失 L_var
        # 计算每个维度的标准差
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2))

        # 3. 协方差损失 L_cov
        # 中心化特征
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        # 协方差矩阵 (D, D)
        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)

        # 提取非对角线元素
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() / D) + (cov_z2[off_diag_mask].pow(2).sum() / D)

        # 总损失
        total_loss = (
            self.lambda_invar * sim_loss +
            self.lambda_var * std_loss +
            self.lambda_cov * cov_loss
        )

        return total_loss, {
            "loss_invar": sim_loss.detach(),
            "loss_var": std_loss.detach(),
            "loss_cov": cov_loss.detach(),
            "loss_vicreg": total_loss.detach()
        }
