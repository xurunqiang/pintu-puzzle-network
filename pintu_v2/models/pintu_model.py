"""
完整“拼图”网络第二版 (PintuModel)
整合中心切片、队列扩展、边缘与语义联合检索、局部融合、全局融合及动量分支
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from pintu_v2.config import ModelConfig
from pintu_v2.data.patch_slicer import PatchSlicer
from pintu_v2.data.patch_pool import PatchPool, BatchInventory
from pintu_v2.models.content_encoder import ContentEncoder
from pintu_v2.models.edge_encoder import EdgeEncoder
from pintu_v2.models.retrieval_scorer import RetrievalScorer
from pintu_v2.models.local_fusion import LocalFusion
from pintu_v2.models.global_fusion import GlobalFusion


class PintuModel(nn.Module):
    """
    “拼图”网络第二版:
    - 仅以中心 Patch 为起点
    - 候选来自 8,192 跨图片候选池，隐藏真实原图与坐标
    - FIFO 队列向外逐步扩展:
        按 “队首中心 -> 补齐空邻居 -> 对中心进行 3x3 局部融合 -> 写回有效位置” 严谨执行，
        完整覆盖全部 g x g (例如 25 个) 位置。
    - 局部十字卷积与局部注意力融合 -> 全网格全局注意力融合 -> 96 维语义表示
    - 动量全图分支提供高质量自监督对齐目标
    """
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.grid_size = self.cfg.grid_size
        self.content_dim = self.cfg.content_dim
        self.edge_dim = self.cfg.edge_dim

        self.slicer = PatchSlicer(grid_size=self.grid_size)
        self.patch_size = self.slicer.patch_size
        self.center_coord = self.slicer.center_coord

        # 核心模块
        self.content_encoder = ContentEncoder(content_dim=self.content_dim)
        self.edge_encoder = EdgeEncoder(edge_dim=self.edge_dim)
        self.scorer = RetrievalScorer(
            pos_penalty_weight=self.cfg.pos_penalty_weight,
            dir_score_weight=0.2,
            edge_gradient_scale=self.cfg.edge_gradient_scale,
            seam_color_weight=self.cfg.seam_color_weight,
            seam_gradient_weight=self.cfg.seam_gradient_weight,
            missing_edge_penalty=self.cfg.missing_edge_penalty,
            seed_anchor_weight=self.cfg.seed_anchor_weight,
            direction_chunk_size=self.cfg.direction_chunk_size
        )
        self.local_fusion = LocalFusion(
            dim=self.content_dim,
            num_heads=self.cfg.local_heads,
            mlp_ratio=self.cfg.local_mlp_ratio
        )
        self.global_fusion = GlobalFusion(
            dim=self.content_dim,
            grid_size=self.grid_size,
            num_layers=self.cfg.global_layers,
            num_heads=self.cfg.global_heads,
            mlp_ratio=self.cfg.global_mlp_ratio
        )

        # 邻居方向与对应边缘索引定义
        # 0: top, 1: right, 2: bottom, 3: left
        self.nbr_offsets = {
            'top': (-1, 0, 2, 0),     # dr, dc, nbr_facing_edge (bottom=2), cand_facing_edge (top=0)
            'right': (0, 1, 3, 1),    # dr, dc, nbr_facing_edge (left=3), cand_facing_edge (right=1)
            'bottom': (1, 0, 0, 2),   # dr, dc, nbr_facing_edge (top=0), cand_facing_edge (bottom=2)
            'left': (0, -1, 1, 3)     # dr, dc, nbr_facing_edge (right=1), cand_facing_edge (left=3)
        }

    def encode_patch(self, patch: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单批次 Patch 提取内容特征、预测位置及 4 个方向的边缘特征与有效性标记。
        输入: patch (B, 3, P, P)
        输出:
            content_feat: (B, 96)
            pred_pos: (B, 2)
            edge_feats: (B, 4, 32) top, right, bottom, left
            edge_val_flags: (B, 4) bool
        """
        B = patch.shape[0]
        if mask is None:
            mask = torch.ones_like(patch[:, :1])
        content_feat, pred_pos = self.content_encoder(patch)

        # 提取 4 个边带
        bands, b_masks = self.slicer.extract_edge_bands(patch, mask)
        edge_list = []
        val_list = []
        for name in ['top', 'right', 'bottom', 'left']:
            e_band = bands[name]
            e_mask = b_masks[name] if b_masks is not None else None
            e_feat, e_val = self.edge_encoder(e_band, e_mask) # e_feat: (B, 32), e_val: (B,)
            edge_list.append(e_feat)
            val_list.append(e_val)

        edge_feats = torch.stack(edge_list, dim=1) # (B, 4, 32)
        edge_val_flags = torch.stack(val_list, dim=1) # (B, 4)
        return content_feat, pred_pos, edge_feats, edge_val_flags

    def encode_full_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        将真实完整图像切分为真实网格，通过内容编码器与全局融合网络输出全图真实语义向量。
        """
        B = img.shape[0]
        patches, masks = self.slicer.slice_image(img) # (B, g, g, 3, P, P)
        g = self.grid_size
        P = self.patch_size

        flat_patches = patches.reshape(B * g * g, 3, P, P)
        content_feats, _ = self.content_encoder(flat_patches) # (B * g * g, 96)

        grid_feats = content_feats.view(B, g, g, self.content_dim).permute(0, 3, 1, 2).contiguous() # (B, 96, g, g)
        z_full = self.global_fusion(grid_feats) # (B, 96)
        return z_full

    def assemble_puzzle(
        self,
        seed_imgs: torch.Tensor,
        seed_img_ids: torch.Tensor,
        pool: PatchPool,
        tau: float = 1.0,
        consumed_inventory: bool = True,
        record_trajectory: bool = True,
        allocation_offset: int = 0,
        allow_self_patches: bool = True,
        inventory_scope: str = "batch"
    ) -> Dict[str, Any]:
        """
        执行完整的“拼图”扩展流程:
        1. 仅提取种子图片的中心 Patch 作为唯一出发点
        2. 初始化网格与批次共享库存管理器
        3. FIFO 队列驱动扩展:
           取出队首中心 -> 补齐所有相邻空邻居 -> 对该中心执行 3x3 局部融合并写回有效位置
           处理完全部 g x g (25) 个网格位置。
        4. 全局网格融合，输出 96 维语义特征与拼接物理拼图
        """
        B = seed_imgs.shape[0]
        g = self.grid_size
        P = self.patch_size
        device = seed_imgs.device
        denom = max(1, g - 1)
        if pool.grid_size != g:
            raise ValueError('候选池与模型网格大小不一致')
        if inventory_scope not in ('batch', 'image'):
            raise ValueError('inventory_scope 必须为 batch 或 image')
        required = (B if inventory_scope == 'batch' else 1) * (g * g - 1)
        if consumed_inventory and pool.pool_size < required:
            raise ValueError(f'{inventory_scope} 库存至少需要 {required} 个候选，当前只有 {pool.pool_size}')

        # 1. 切片获取种子图的中心块
        seed_patches, seed_masks = self.slicer.slice_image(seed_imgs)
        cr, cc = self.center_coord
        center_patch = seed_patches[:, cr, cc] # (B, 3, P, P)
        center_mask = seed_masks[:, cr, cc]   # (B, 1, P, P)

        # 编码中心块
        center_feat, _, center_edge_feats, center_edge_vals = self.encode_patch(center_patch, center_mask)
        # center_feat: (B, 96), center_edge_feats: (B, 4, 32), center_edge_vals: (B, 4)

        # 计算中心块边带法向梯度强度 (用于 S_edge 动态权重)
        center_bands, center_band_masks = self.slicer.extract_edge_bands(center_patch, center_mask)
        center_edge_grads = torch.zeros((B, 4), device=device)
        for e_idx, name in enumerate(['top', 'right', 'bottom', 'left']):
            center_edge_grads[:, e_idx] = center_bands[name][:, 3:6, 0, :].abs().mean(dim=(1, 2))

        # 2. 初始化网格状态
        grid_features = torch.zeros((B, self.content_dim, g, g), device=device)
        # 方向预测始终比较未经上下文融合的块特征。
        grid_raw_features = torch.zeros_like(grid_features)
        grid_mask = torch.zeros((B, 1, g, g), device=device)
        grid_edge_feats = torch.zeros((B, 4, self.edge_dim, g, g), device=device)
        grid_edge_vals = torch.zeros((B, 4, g, g), dtype=torch.bool, device=device)
        grid_edge_grads = torch.zeros((B, 4, g, g), device=device)
        assembled_patches = torch.zeros((B, g, g, 3, P, P), device=device)
        assembled_masks = torch.zeros((B, g, g, 1, P, P), device=device)
        grid_bands = torch.zeros((B, 4, 6, 2, P, g, g), device=device)
        grid_band_masks = torch.zeros((B, 4, 1, 2, P, g, g), device=device)

        # 放入中心块
        grid_features[:, :, cr, cc] = center_feat
        grid_raw_features[:, :, cr, cc] = center_feat
        grid_mask[:, :, cr, cc] = 1.0
        grid_edge_feats[:, :, :, cr, cc] = center_edge_feats
        grid_edge_vals[:, :, cr, cc] = center_edge_vals
        grid_edge_grads[:, :, cr, cc] = center_edge_grads
        assembled_patches[:, cr, cc] = center_patch
        assembled_masks[:, cr, cc] = center_mask
        for e_idx, name in enumerate(['top', 'right', 'bottom', 'left']):
            grid_bands[:, e_idx, :, :, :, cr, cc] = center_bands[name].detach()
            grid_band_masks[:, e_idx, :, :, :, cr, cc] = center_band_masks[name]

        # 3. 初始化库存管理器 (批次共享库存)
        inventory = BatchInventory(
            batch_size=B,
            pool_size=pool.pool_size,
            source_img_ids=pool.source_img_ids,
            seed_img_ids=seed_img_ids,
            consumed_inventory=consumed_inventory,
            allow_self_patches=allow_self_patches,
            device=device,
            inventory_scope=inventory_scope
        )

        # 4. FIFO 队列驱动逐步扩展
        queue = deque([(cr, cc)])
        queued_set = {(cr, cc)}
        trajectory = []
        fusion_count = 0

        # 获取候选池缓存特征
        pool_c_feats = pool.content_feats       # (K, 96)
        pool_e_feats = pool.edge_feats          # (K, 4, 32)
        pool_e_vals = pool.edge_valid_flags     # (K, 4) bool
        pool_pred_pos = pool.pred_positions     # (K, 2)
        # 同一次组装期间参数固定；归一化与方向头候选投影只计算一次。
        pool_cache = {
            'normalized_content': F.normalize(pool_c_feats, dim=-1),
            'direction_hidden': F.linear(pool_c_feats, self.content_encoder.dir_head[0].weight[:, self.content_dim:]),
        }

        step_idx = 0
        while queue:
            curr_r, curr_c = queue.popleft()

            # --- 步骤 A: 检查并补齐当前中心四周的所有空邻居 ---
            for dir_name, (dr, dc, nbr_facing_edge, cand_facing_edge) in self.nbr_offsets.items():
                nr, nc = curr_r + dr, curr_c + dc

                # 越界检查
                if not (0 <= nr < g and 0 <= nc < g):
                    continue

                # 若邻居位置已经填入，则跳过检索
                if (nr, nc) in queued_set:
                    continue

                # 检查库存中是否有可用合法候选
                # 合法性由评分器与分配器检查；空间占用由 Python 集合维护。

                target_norm_coord = torch.tensor([nr / denom, nc / denom], device=device, dtype=torch.float32)

                # 收集目标空槽位 (nr, nc) 的所有已填入正交邻居
                nbr_c_list = []
                nbr_raw_list = []
                nbr_bands_dict, nbr_masks_dict = {}, {}
                nbr_edges_dict_batch = [{} for _ in range(B)]
                nbr_val_dict_batch = [{} for _ in range(B)]
                nbr_grad_list = []

                for check_dir, (cdr, cdc, c_nbr_facing, c_cand_facing) in self.nbr_offsets.items():
                    adj_r, adj_c = nr + cdr, nc + cdc
                    if 0 <= adj_r < g and 0 <= adj_c < g:
                        if (adj_r, adj_c) in queued_set:
                            nbr_c = grid_features[:, :, adj_r, adj_c] # (B, 96)
                            nbr_c_list.append(nbr_c)
                            nbr_raw_list.append(grid_raw_features[:, :, adj_r, adj_c])
                            nbr_bands_dict[check_dir] = grid_bands[:, c_nbr_facing, :, :, :, adj_r, adj_c]
                            nbr_masks_dict[check_dir] = grid_band_masks[:, c_nbr_facing, :, :, :, adj_r, adj_c]

                            for b in range(B):
                                nbr_edges_dict_batch[b][check_dir] = grid_edge_feats[b, c_nbr_facing, :, adj_r, adj_c]
                                nbr_val_dict_batch[b][check_dir] = grid_edge_vals[b, c_nbr_facing, adj_r, adj_c]

                            nbr_grad_list.append(grid_edge_grads[:, c_nbr_facing, adj_r, adj_c]) # (B,)

                nbr_content_feats = torch.stack(nbr_c_list, dim=1) # (B, M, 96)
                nbr_grads = torch.stack(nbr_grad_list, dim=1).mean(dim=1) # (B,)

                # 获取可用候选掩码
                valid_mask = inventory.get_valid_mask() # (B, K)

                # 计算综合匹配评分 S(p, t)，接入方向头预测与无效边带过滤
                scores = self.scorer.compute_scores(
                    target_norm_coord=target_norm_coord,
                    nbr_content_feats=nbr_content_feats,
                    nbr_edge_feats=nbr_edges_dict_batch,
                    nbr_edge_val_flags=nbr_val_dict_batch,
                    nbr_grad_magnitudes=nbr_grads,
                    pool_content_feats=pool_c_feats,
                    pool_edge_feats=pool_e_feats,
                    pool_edge_val_flags=pool_e_vals,
                    pool_pred_positions=pool_pred_pos,
                    valid_mask=valid_mask,
                    content_encoder=self.content_encoder,
                    nbr_raw_content_feats=torch.stack(nbr_raw_list, dim=1),
                    seed_content_feat=center_feat,
                    nbr_bands=nbr_bands_dict, nbr_masks=nbr_masks_dict,
                    pool_bands=pool.edge_bands, pool_masks=pool.edge_masks,
                    required_edges=[edge for edge, (dr_, dc_) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)])
                                    if 0 <= nr + dr_ < g and 0 <= nc + dc_ < g],
                    pool_cache=pool_cache
                ) # (B, K)
                if not consumed_inventory:
                    scores = scores - self.cfg.repetition_score_weight * torch.log1p(inventory.hard_usage_counts)

                # Straight-Through 选块并仲裁批次内冲突
                sel_content, sel_idx, soft_w = self.scorer.select_with_straight_through(
                    scores=scores,
                    pool_features=pool_c_feats,
                    valid_mask=valid_mask,
                    tau=tau,
                    consumed_inventory=consumed_inventory and inventory_scope == 'batch',
                    allocation_offset=allocation_offset + step_idx
                )

                # 更新批次库存消耗与软概率累加
                inventory.consume(sel_idx)
                if not consumed_inventory:
                    inventory.record_soft_usage(soft_w)

                # 记录物理选中的真实 Patch 像素与边带
                chosen_patches = pool.patches[sel_idx]       # (B, 3, P, P)
                chosen_edge_feats = pool.edge_feats[sel_idx] # (B, 4, 32)
                chosen_edge_vals = pool.edge_valid_flags[sel_idx] # (B, 4) bool

                # 计算选中 Patch 边带的法向梯度
                chosen_bands = pool.edge_bands[sel_idx]      # (B, 4, 6, 2, P)
                chosen_grads = chosen_bands[:, :, 3:6, 0, :].abs().mean(dim=(2, 3)) # (B, 4)

                # 填入网格对应位置
                grid_features[:, :, nr, nc] = sel_content
                grid_raw_features = grid_raw_features.clone()
                grid_raw_features[:, :, nr, nc] = sel_content
                grid_mask[:, :, nr, nc] = 1.0
                grid_edge_feats[:, :, :, nr, nc] = chosen_edge_feats
                grid_edge_vals[:, :, nr, nc] = chosen_edge_vals
                grid_edge_grads[:, :, nr, nc] = chosen_grads
                assembled_patches[:, nr, nc] = chosen_patches
                assembled_masks[:, nr, nc] = pool.patch_masks[sel_idx]
                grid_bands[:, :, :, :, :, nr, nc] = chosen_bands
                grid_band_masks[:, :, :, :, :, nr, nc] = pool.edge_masks[sel_idx]

                # 将新填补的位置加入队列等待作为后续中心扩展
                if (nr, nc) not in queued_set:
                    queue.append((nr, nc))
                    queued_set.add((nr, nc))

                # 记录挑选轨迹
                if record_trajectory:
                    top_scores, top_indices = scores.detach().topk(min(5, pool.pool_size), dim=-1)
                    trajectory.append({
                        "step": step_idx,
                        "target_coord": (nr, nc),
                        "selected_idx": sel_idx.detach().cpu(),
                        "scores": top_scores.cpu(),
                        "candidate_indices": top_indices.cpu(),
                    })
                step_idx += 1

            # --- 步骤 B: 对当前队首中心执行 3x3 局部融合并写回 ---
            # 此时 (curr_r, curr_c) 的所有四周有效邻居均已就位
            fused = self.local_fusion(grid_features, grid_mask, curr_r, curr_c,
                                      return_all=True, valid_coords=queued_set)
            grid_features = grid_features.clone()
            for (r, c), features in fused.items():
                grid_features[:, :, r, c] = features
            fusion_count += 1

        # 5. 全网格填充完毕，进入全局融合网络
        z_puzzle = self.global_fusion(grid_features) # (B, 96)

        # 6. 拼装还原物理图像 (B, 3, H_padded, W_padded)
        full_puzzle_img = assembled_patches.permute(0, 3, 1, 4, 2, 5).reshape(
            B, 3, g * P, g * P
        )

        return {
            "z_puzzle": z_puzzle,
            "puzzle_img": full_puzzle_img,
            "assembled_patches": assembled_patches,
            "assembled_masks": assembled_masks,
            "grid_features": grid_features,
            "trajectory": trajectory,
            "usage_counts": inventory.hard_usage_counts,
            "repetition_loss": inventory.get_repetition_loss(),
            "fusion_count": fusion_count,
            "inventory_scope": inventory_scope,
            "unique_patch_counts": (inventory.hard_usage_counts > 0).sum(dim=1),
            "hard_repeat_rate": 1 - (inventory.hard_usage_counts > 0).sum(dim=1).float() / (g * g - 1),
        }
