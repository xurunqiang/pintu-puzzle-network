"""
检索评分器 (RetrievalScorer) 与直通梯度估计 (Straight-Through Estimator)
支持方向预测打分接入、全无效边带排除与批次冲突仲裁
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RetrievalScorer(nn.Module):
    """
    负责计算目标网格空槽位 t 与候选池中各 Patch p 的综合匹配评分 S(p, t)，
    并通过 Straight-Through Softmax 实现离散硬选与连续梯度的统一。
    
    综合评分公式:
        S = 语义 + 边缘 + 方向 + 多邻居一致性 - 位置误差 - 物理接缝误差 - 缺失边惩罚
    其中:
        w_t = 0.1 + 0.7 * (1 - exp(-g_t / scale))
    """
    def __init__(self, pos_penalty_weight: float = 0.1, dir_score_weight: float = 0.2,
                 edge_gradient_scale: float = 0.1, seam_color_weight: float = 0.25,
                 seam_gradient_weight: float = 0.15, missing_edge_penalty: float = 0.5,
                 seed_anchor_weight: float = 0.2, direction_chunk_size: int = 512):
        super().__init__()
        self.pos_penalty_weight = pos_penalty_weight
        self.dir_score_weight = dir_score_weight
        if edge_gradient_scale <= 0 or direction_chunk_size < 1:
            raise ValueError('梯度尺度和方向分块大小必须为正数')
        if not 0 <= seed_anchor_weight <= 1:
            raise ValueError('seed_anchor_weight 必须在 [0, 1]')
        self.edge_gradient_scale = edge_gradient_scale
        self.seam_color_weight = seam_color_weight
        self.seam_gradient_weight = seam_gradient_weight
        self.missing_edge_penalty = missing_edge_penalty
        self.seed_anchor_weight = seed_anchor_weight
        self.direction_chunk_size = direction_chunk_size

        # 方向映射:
        # neighbor_dir: 'top' -> 邻居在 t 上方 -> 邻居的 bottom (2) 面对 t，候选的 top (0) 面对邻居，t 相对于邻居的方向为 bottom (2)
        # neighbor_dir: 'right' -> 邻居在 t 右侧 -> 邻居的 left (3) 面对 t，候选的 right (1) 面对邻居，t 相对于邻居的方向为 left (3)
        # neighbor_dir: 'bottom' -> 邻居在 t 下方 -> 邻居的 top (0) 面对 t，候选的 bottom (2) 面对邻居，t 相对于邻居的方向为 top (0)
        # neighbor_dir: 'left' -> 邻居在 t 左侧 -> 邻居的 right (1) 面对 t，候选的 left (3) 面对邻居，t 相对于邻居的方向为 right (1)
        self.dir_mapping = {
            'top': {'cand_edge': 0, 'nbr_edge': 2, 'rel_dir_label': 2},
            'right': {'cand_edge': 1, 'nbr_edge': 3, 'rel_dir_label': 3},
            'bottom': {'cand_edge': 2, 'nbr_edge': 0, 'rel_dir_label': 0},
            'left': {'cand_edge': 3, 'nbr_edge': 1, 'rel_dir_label': 1}
        }

    def compute_scores(
        self,
        target_norm_coord: torch.Tensor,                    # (2,)
        nbr_content_feats: torch.Tensor,                    # (B, M, 96) M 为已有有效邻居数
        nbr_edge_feats: List[Dict[str, torch.Tensor]],      # 长度为 B 的列表，每项记录 neighbor_dir -> (32,)
        nbr_edge_val_flags: List[Dict[str, bool]],          # 长度为 B 的列表，每项记录 neighbor_dir -> bool (是否有效)
        nbr_grad_magnitudes: torch.Tensor,                  # (B,) 已有邻居接缝处的平均法向梯度强度 g_t
        pool_content_feats: torch.Tensor,                   # (K, 96)
        pool_edge_feats: torch.Tensor,                      # (K, 4, 32)
        pool_edge_val_flags: torch.Tensor,                  # (K, 4) bool
        pool_pred_positions: torch.Tensor,                  # (K, 2)
        valid_mask: torch.Tensor,                           # (B, K) 可用候选掩码
        content_encoder: Optional[nn.Module] = None,
        nbr_raw_content_feats: Optional[torch.Tensor] = None,
        seed_content_feat: Optional[torch.Tensor] = None,
        nbr_bands: Optional[Dict[str, torch.Tensor]] = None,
        nbr_masks: Optional[Dict[str, torch.Tensor]] = None,
        pool_bands: Optional[torch.Tensor] = None,
        pool_masks: Optional[torch.Tensor] = None,
        required_edges: Optional[List[int]] = None,
        pool_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        批量计算 S(p, t)。
        输出:
            scores: (B, K)
        """
        B, K = nbr_content_feats.shape[0], pool_content_feats.shape[0]
        if (~valid_mask.any(dim=1)).any():
            raise RuntimeError("候选池库存耗尽：存在无合法候选的样本")
        query = F.normalize(nbr_content_feats.mean(dim=1), dim=-1)
        if seed_content_feat is not None:
            query = F.normalize((1 - self.seed_anchor_weight) * query +
                                self.seed_anchor_weight * F.normalize(seed_content_feat, dim=-1), dim=-1)
        normalized_pool = (pool_cache['normalized_content'] if pool_cache is not None
                           else F.normalize(pool_content_feats, dim=-1))
        s_sem = query @ normalized_pool.t()
        seam_cost = torch.zeros_like(s_sem)
        edge_sum = torch.zeros_like(s_sem)
        edge_count = torch.zeros_like(s_sem)
        edge_min = torch.full_like(s_sem, float('inf'))
        s_dir = torch.zeros_like(s_sem)
        dir_count = torch.zeros(B, 1, device=s_sem.device, dtype=s_sem.dtype)

        if content_encoder is not None:
            first = content_encoder.dir_head[0]
            dim = pool_content_feats.shape[-1]
            w_a, w_b = first.weight[:, :dim], first.weight[:, dim:]
            h_pool = (pool_cache['direction_hidden'] if pool_cache is not None
                      else F.linear(pool_content_feats, w_b))
            last = content_encoder.dir_head[2]

        # 每个方向使用批量矩阵运算，不再逐样本 launch 矩阵向量内核。
        for direction, spec in self.dir_mapping.items():
            rows = [b for b in range(B) if direction in nbr_edge_feats[b]]
            if not rows:
                continue
            row_idx = torch.tensor(rows, device=s_sem.device)
            e = torch.stack([nbr_edge_feats[b][direction] for b in rows])
            flags = torch.stack([torch.as_tensor(nbr_edge_val_flags[b].get(direction, True),
                                 device=s_sem.device, dtype=torch.bool) for b in rows])
            candidates = pool_edge_feats[:, spec['cand_edge']]
            similarities = e @ candidates.t()
            pair_valid = flags[:, None] & pool_edge_val_flags[:, spec['cand_edge']][None, :]
            edge_sum = edge_sum.index_add(0, row_idx, similarities * pair_valid)
            edge_count = edge_count.index_add(0, row_idx, pair_valid.to(s_sem.dtype))
            # index_copy 创建新张量，避免改写反向传播需要的旧值。
            edge_min = edge_min.index_copy(0, row_idx, torch.minimum(
                edge_min[row_idx], similarities.masked_fill(~pair_valid, float('inf'))))

            if nbr_bands is not None and direction in nbr_bands:
                # 物理误差没有可学习参数，分块计算且不保留跨 BFS 的像素计算图。
                cost = self._seam_cost(nbr_bands[direction][row_idx], nbr_masks[direction][row_idx],
                                       pool_bands[:, spec['cand_edge']], pool_masks[:, spec['cand_edge']])
                seam_cost = seam_cost.index_add(0, row_idx, cost)

            if content_encoder is not None:
                raw_features = nbr_content_feats if nbr_raw_content_feats is None else nbr_raw_content_feats
                z = torch.stack([raw_features[b, list(nbr_edge_feats[b]).index(direction)]
                                 for b in rows])
                h_nbr = F.linear(z, w_a, first.bias)
                chunks = []
                for start in range(0, K, self.direction_chunk_size):
                    candidate = h_pool[start:start + self.direction_chunk_size]
                    inputs = (h_nbr, candidate, last.weight, last.bias, spec['rel_dir_label'])
                    # 方向头的 B×K×hidden 激活跨整个 BFS 保留会占用大量显存。
                    if torch.is_grad_enabled():
                        log_p = checkpoint(self._direction_log_prob, *inputs, use_reentrant=False)
                    else:
                        log_p = self._direction_log_prob(*inputs)
                    chunks.append(log_p)
                s_dir = s_dir.index_add(0, row_idx, torch.cat(chunks, dim=1))
                dir_count = dir_count.index_add(0, row_idx, torch.ones(len(rows), 1, device=s_sem.device))

        s_edge = edge_sum / edge_count.clamp_min(1)
        s_adj = 0.1 * torch.where(edge_count > 1, edge_min, torch.zeros_like(edge_min))
        s_dir = s_dir / dir_count.clamp_min(1)
        w_t = (0.1 + 0.7 * (-torch.expm1(-nbr_grad_magnitudes.clamp_min(0) /
                                        self.edge_gradient_scale)))[:, None]
        # 没有真实接缝证据的候选，使用完整语义权重，不用零边缘伪装成有效匹配。
        w_t = w_t * (edge_count > 0)
        distance = (pool_pred_positions - target_norm_coord[None]).square().sum(dim=-1)
        scores = ((1 - w_t) * s_sem + w_t * s_edge + self.dir_score_weight * s_dir
                  + s_adj - self.pos_penalty_weight * distance[None]
                  - seam_cost / max(1, nbr_content_feats.shape[1]))
        if required_edges:
            # 同时覆盖未来要接邻居的方向，避免把源图人工填充边无代价地放到内部。
            coverage = (pool_masks[:, required_edges, 0, 0].mean(dim=-1)
                        if pool_masks is not None else pool_edge_val_flags[:, required_edges].float())
            scores = scores - self.missing_edge_penalty * (1 - coverage.mean(dim=1))[None]
        return scores.masked_fill(~valid_mask, float('-inf'))

    @torch.no_grad()
    def _seam_cost(self, neighbor, neighbor_mask, candidates, candidate_masks):
        """规范边带的颜色差及反向法线梯度差；仅使用两侧有效像素。"""
        costs = []
        for start in range(0, candidates.shape[0], self.direction_chunk_size):
            band = candidates[start:start + self.direction_chunk_size]
            mask = candidate_masks[start:start + self.direction_chunk_size]
            valid = neighbor_mask[:, None, 0, 0] * mask[None, :, 0, 0]
            color = (neighbor[:, None, :3, 0] - band[None, :, :3, 0]).abs()
            color = (color * valid[:, :, None]).sum(dim=(-1, -2)) / (3 * valid.sum(-1)).clamp_min(1)
            grad_valid = valid * neighbor_mask[:, None, 0, 1] * mask[None, :, 0, 1]
            gradient = (neighbor[:, None, 3:6, 0] + band[None, :, 3:6, 0]).abs()
            gradient = ((gradient * grad_valid[:, :, None]).sum(dim=(-1, -2)) /
                        (3 * grad_valid.sum(-1)).clamp_min(1))
            costs.append(self.seam_color_weight * color + self.seam_gradient_weight * gradient)
        return torch.cat(costs, dim=1)

    @staticmethod
    def _direction_log_prob(neighbor, candidate, weight, bias, label):
        logits = F.linear(F.gelu(neighbor[:, None, :] + candidate[None, :, :]), weight, bias)
        return F.log_softmax(logits, dim=-1)[..., label]

    def select_with_straight_through(
        self,
        scores: torch.Tensor,
        pool_features: torch.Tensor,
        valid_mask: torch.Tensor,
        tau: float = 1.0,
        consumed_inventory: bool = True,
        allocation_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Straight-Through 选块与梯度传递，具备批次冲突仲裁功能。
        若 consumed_inventory=True，同一步内各样本不可挑选同一候选块。
        """
        B, K = scores.shape
        if valid_mask.shape != scores.shape or (~valid_mask.any(dim=1)).any():
            raise RuntimeError("候选池库存耗尽或合法掩码不匹配")
        if not torch.isfinite(scores[valid_mask]).all():
            raise FloatingPointError("合法候选分数包含 NaN 或 Inf")
        masked_scores = scores.masked_fill(~valid_mask, float('-inf'))
        selection_mask = valid_mask.clone()
        if consumed_inventory:
            # 每行前 B 个合法候选足以寻找一个大小为 B 的二分匹配。
            # 仅取前 B 个候选，避免每一步对整个池做完整排序。
            # 同分候选沿用 torch.topk 的次序，不承诺最小 ID 优先。
            order = torch.topk(masked_scores.detach(), min(B, K), dim=1, sorted=True).indices
            legal = valid_mask.gather(1, order).cpu().tolist()
            preferences = [[candidate for candidate, ok in zip(row, flags) if ok]
                           for row, flags in zip(order.cpu().tolist(), legal)]
            owners, chosen = {}, [-1] * B

            def assign(row, visited):
                # 先保留其他样本的分配；只有无空闲候选时才用增广路径修复冲突。
                for candidate in preferences[row]:
                    if candidate not in visited and candidate not in owners:
                        owners[candidate], chosen[row] = row, candidate
                        return True
                for candidate in preferences[row]:
                    if candidate in visited:
                        continue
                    visited.add(candidate)
                    if assign(owners[candidate], visited):
                        owners[candidate], chosen[row] = row, candidate
                        return True
                return False

            for offset in range(B):
                row = (offset + allocation_offset) % B
                if not assign(row, set()):
                    raise RuntimeError(f"候选池无法满足批次的无冲突分配，样本 {row} 无可行匹配")
            selected_indices = torch.tensor(chosen, device=scores.device)
            selection_mask[:, selected_indices] = False
            selection_mask[torch.arange(B, device=scores.device), selected_indices] = True
        else:
            selected_indices = masked_scores.argmax(dim=-1)

        # 软反向分布必须与最终分配的库存一致，不能把梯度发给其他样本已领走的块。
        soft_weights = F.softmax(masked_scores.masked_fill(~selection_mask, float('-inf')) /
                                 max(tau, 1e-3), dim=-1)
        hard_weights = F.one_hot(selected_indices, num_classes=K).to(soft_weights.dtype)
        st_weights = (hard_weights - soft_weights).detach() + soft_weights
        return st_weights @ pool_features, selected_indices, soft_weights
