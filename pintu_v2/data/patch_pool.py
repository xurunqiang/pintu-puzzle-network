"""
跨图片候选块池 (PatchPool) 与批次共享库存管理器 (BatchInventory)
"""

import math
import copy
from typing import Dict, Tuple, Optional, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from pintu_v2.data.patch_slicer import PatchSlicer


class PatchPool:
    """
    管理 8,192 个唯一非中心候选 Patch 的物理像素、边带特征及动量特征缓存。
    持久基础池来自训练集；临时混合池还包含当前种子图的全部非中心块。
    """
    def __init__(
        self,
        pool_size: int = 8192,
        grid_size: int = 5,
        device: torch.device = torch.device("cpu"),
        seed: int = 42
    ):
        self.pool_size = pool_size
        self.grid_size = grid_size
        self.device = device
        self.seed = seed
        self.slicer = PatchSlicer(grid_size=grid_size)
        self.patch_size = self.slicer.patch_size
        self.center_coord = self.slicer.center_coord

        # 边缘方向索引定义: 0: top, 1: right, 2: bottom, 3: left
        self.edge_names = ['top', 'right', 'bottom', 'left']
        self.opposing_edge_idx = {0: 2, 1: 3, 2: 0, 3: 1} # top<->bottom, right<->left

        # 候选池张量存储
        self.patches: Optional[torch.Tensor] = None           # (pool_size, 3, P, P)
        self.patch_masks: Optional[torch.Tensor] = None       # (pool_size, 1, P, P)
        self.edge_bands: Optional[torch.Tensor] = None        # (pool_size, 4, 6, 2, P)
        self.edge_masks: Optional[torch.Tensor] = None        # (pool_size, 4, 1, 2, P)
        self.edge_valid_flags: Optional[torch.Tensor] = None  # (pool_size, 4) bool
        self.source_img_ids: Optional[torch.Tensor] = None    # (pool_size,) 真实底层图片 ID
        self.source_coords: Optional[torch.Tensor] = None     # (pool_size, 2) in [0, 1]
        self.sampled_tuples: List[Tuple[int, int, int, int]] = [] # (sub_idx, img_id, r, c)

        # 动态特征缓存
        self.content_feats: Optional[torch.Tensor] = None     # (pool_size, 96)
        self.edge_feats: Optional[torch.Tensor] = None        # (pool_size, 4, 32)
        self.pred_positions: Optional[torch.Tensor] = None    # (pool_size, 2)
        self._content_snapshot = None
        self._edge_snapshot = None

    def build_pool(self, train_dataset):
        """
        从真实的训练集 (如 train_subset) 中均匀抽取 pool_size 个唯一非中心 Patch。
        train_dataset 可以是 Subset 或 Dataset，使用 len(train_dataset) 与索引直接访问。
        """
        num_source_images = len(train_dataset)
        g = torch.Generator().manual_seed(self.seed)

        # 1. 确定所有合法的非中心位置
        non_center_coords = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) != self.center_coord:
                    non_center_coords.append((r, c))

        num_non_center = len(non_center_coords)
        total_available = num_source_images * num_non_center
        assert self.pool_size <= total_available, f"候选池大小 {self.pool_size} 超过可用总数 {total_available}"

        # 2. 均匀无放回采样 pool_size 个索引
        sampled_flat_indices = torch.randperm(total_available, generator=g)[:self.pool_size]

        # 3. 组织采样列表: (sub_idx, r, c)
        sample_tuples = []
        for flat_idx in sampled_flat_indices.tolist():
            sub_idx = flat_idx // num_non_center
            coord_idx = flat_idx % num_non_center
            r, c = non_center_coords[coord_idx]
            sample_tuples.append((sub_idx, r, c))

        # 按样本分组以避免重复切片同一张图片
        sub_to_slots: Dict[int, list] = {}
        for pool_idx, (sub_idx, r, c) in enumerate(sample_tuples):
            if sub_idx not in sub_to_slots:
                sub_to_slots[sub_idx] = []
            sub_to_slots[sub_idx].append((pool_idx, r, c))

        # 4. 初始化容器
        P = self.patch_size
        patches_buf = torch.zeros((self.pool_size, 3, P, P), dtype=torch.float32)
        masks_buf = torch.zeros((self.pool_size, 1, P, P), dtype=torch.float32)
        edge_bands_buf = torch.zeros((self.pool_size, 4, 6, 2, P), dtype=torch.float32)
        edge_masks_buf = torch.zeros((self.pool_size, 4, 1, 2, P), dtype=torch.float32)
        edge_valid_buf = torch.zeros((self.pool_size, 4), dtype=torch.bool)
        source_ids_buf = torch.zeros((self.pool_size,), dtype=torch.long)
        source_coords_buf = torch.zeros((self.pool_size, 2), dtype=torch.float32)
        self.sampled_tuples = [None] * self.pool_size

        # 5. 遍历切片提取
        denom = max(1, self.grid_size - 1)
        for sub_idx, slots in sub_to_slots.items():
            item = train_dataset[sub_idx]
            raw_img = item["raw_img"].unsqueeze(0) # (1, 3, 32, 32)
            real_img_id = item["img_id"]

            img_patches, img_masks = self.slicer.slice_image(raw_img)
            # img_patches: (1, g, g, 3, P, P)

            for pool_idx, r, c in slots:
                p = img_patches[0, r, c] # (3, P, P)
                m = img_masks[0, r, c]   # (1, P, P)

                bands, b_masks = self.slicer.extract_edge_bands(p.unsqueeze(0), m.unsqueeze(0))

                patches_buf[pool_idx] = p
                masks_buf[pool_idx] = m
                for e_idx, name in enumerate(self.edge_names):
                    b = bands[name][0]
                    bm = b_masks[name][0]
                    edge_bands_buf[pool_idx, e_idx] = b
                    edge_masks_buf[pool_idx, e_idx] = bm
                    # 外边界有效像素数是否 > 0
                    edge_valid_buf[pool_idx, e_idx] = (bm[0, 0, :] > 0.5).any()

                source_ids_buf[pool_idx] = real_img_id
                source_coords_buf[pool_idx] = torch.tensor([r / denom, c / denom], dtype=torch.float32)
                self.sampled_tuples[pool_idx] = (sub_idx, int(real_img_id), r, c)

        self.patches = patches_buf.to(self.device)
        self.patch_masks = masks_buf.to(self.device)
        self.edge_bands = edge_bands_buf.to(self.device)
        self.edge_masks = edge_masks_buf.to(self.device)
        self.edge_valid_flags = edge_valid_buf.to(self.device)
        self.source_img_ids = source_ids_buf.to(self.device)
        self.source_coords = source_coords_buf.to(self.device)

    def to(self, device: torch.device):
        self.device = device
        if self.patches is not None:
            self.patches = self.patches.to(device)
            self.patch_masks = self.patch_masks.to(device)
            self.edge_bands = self.edge_bands.to(device)
            self.edge_masks = self.edge_masks.to(device)
            self.edge_valid_flags = self.edge_valid_flags.to(device)
            self.source_img_ids = self.source_img_ids.to(device)
            self.source_coords = self.source_coords.to(device)
        if self.content_feats is not None:
            self.content_feats = self.content_feats.to(device)
            self.edge_feats = self.edge_feats.to(device)
            self.pred_positions = self.pred_positions.to(device)
        if self._content_snapshot is not None:
            self._content_snapshot.to(device)
            self._edge_snapshot.to(device)
        return self

    @torch.no_grad()
    def refresh_features(self, content_encoder: nn.Module, edge_encoder: nn.Module, batch_size: int = 512,
                         capture_snapshot: bool = True):
        """
        使用当前 (或动量) 编码器刷新池中所有候选 Patch 的内容特征、边缘特征与位置预测。
        """
        assert self.patches is not None, "候选池未构建，请先调用 build_pool"
        content_training, edge_training = content_encoder.training, edge_encoder.training
        content_encoder.eval()
        edge_encoder.eval()

        all_content_feats = []
        all_pred_pos = []
        all_edge_feats = []

        N = self.pool_size
        for i in range(0, N, batch_size):
            p_batch = self.patches[i:i+batch_size] # (B, 3, P, P)
            m_batch = self.patch_masks[i:i+batch_size]

            c_feat, pred_pos = content_encoder(p_batch)
            all_content_feats.append(c_feat)
            all_pred_pos.append(pred_pos)

            e_bands = self.edge_bands[i:i+batch_size]
            e_masks = self.edge_masks[i:i+batch_size]
            B_cur = p_batch.shape[0]

            flat_bands = e_bands.reshape(B_cur * 4, 6, 2, self.patch_size)
            flat_masks = e_masks.reshape(B_cur * 4, 1, 2, self.patch_size)
            e_feat, _ = edge_encoder(flat_bands, flat_masks) # (B_cur * 4, 32)
            all_edge_feats.append(e_feat.reshape(B_cur, 4, 32))

        self.content_feats = torch.cat(all_content_feats, dim=0) # (K, 96)
        self.pred_positions = torch.cat(all_pred_pos, dim=0)    # (K, 2)
        self.edge_feats = torch.cat(all_edge_feats, dim=0)        # (K, 4, 32)
        content_encoder.train(content_training)
        edge_encoder.train(edge_training)
        # 注入新碎片时使用生成基础缓存的同一套编码器，避免来源特征空间不一致。
        if capture_snapshot:
            self._content_snapshot = copy.deepcopy(content_encoder).eval().requires_grad_(False)
            self._edge_snapshot = copy.deepcopy(edge_encoder).eval().requires_grad_(False)

    @torch.no_grad()
    def with_source_patches(self, images, image_ids):
        """构造临时候选池：本批每张图全部非中心块 + 跨图干扰块，总数保持 K。

        不修改基础池；ID/源坐标仅用于建池去重和诊断，不提供给评分器。
        images 必须与种子来自同一增强视图，不能混用增强中心与原始碎片。
        """
        if self._content_snapshot is None or self._edge_snapshot is None:
            raise ValueError('候选池缺少特征编码器快照，请先刷新缓存或恢复完整检查点')
        if images.shape[0] != image_ids.numel() or image_ids.unique().numel() != image_ids.numel():
            raise ValueError('每个种子需要唯一图片 ID；重复图片请分批组装')
        g, p = self.grid_size, self.patch_size
        coords = [(r, c) for r in range(g) for c in range(g) if (r, c) != self.center_coord]
        count = images.shape[0] * len(coords)
        if count > self.pool_size:
            raise ValueError(f'保证原图碎片需要 {count} 个候选槽位，当前池只有 {self.pool_size}')
        keep = (~torch.isin(self.source_img_ids, image_ids.to(self.device))).nonzero().flatten()
        keep = keep[:self.pool_size - count]
        if keep.numel() != self.pool_size - count:
            raise ValueError('跨图干扰块不足以补齐固定候选池，请重新构建基础池')
        patches, masks = self.slicer.slice_image(images)
        rows, cols = zip(*coords)
        injected = PatchPool(count, g, device=self.device, seed=self.seed)
        injected.patches = patches[:, list(rows), list(cols)].reshape(count, 3, p, p).detach()
        injected.patch_masks = masks[:, list(rows), list(cols)].reshape(count, 1, p, p)
        bands, band_masks = self.slicer.extract_edge_bands(injected.patches, injected.patch_masks)
        injected.edge_bands = torch.stack([bands[n] for n in self.edge_names], dim=1)
        injected.edge_masks = torch.stack([band_masks[n] for n in self.edge_names], dim=1)
        injected.edge_valid_flags = (injected.edge_masks[:, :, 0, 0] > 0.5).any(dim=-1)
        injected.source_img_ids = image_ids.to(self.device).repeat_interleave(len(coords))
        injected.source_coords = images.new_tensor(coords).repeat(images.shape[0], 1) / (g - 1)
        injected.refresh_features(self._content_snapshot, self._edge_snapshot, capture_snapshot=False)
        mixed = PatchPool(self.pool_size, g, device=self.device, seed=self.seed)
        # 固定种子打乱槽位，消除“前面是答案块”及网格顺序的提示。
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        order = torch.randperm(self.pool_size, generator=generator, device=self.device)
        for name in ('patches', 'patch_masks', 'edge_bands', 'edge_masks', 'edge_valid_flags',
                     'source_img_ids', 'source_coords', 'content_feats', 'edge_feats', 'pred_positions'):
            value = torch.cat((getattr(injected, name), getattr(self, name)[keep]), dim=0)
            setattr(mixed, name, value[order])
        mixed._content_snapshot, mixed._edge_snapshot = self._content_snapshot, self._edge_snapshot
        return mixed

    def state_dict(self):
        """保存实际用于检索的像素与缓存，而非仅保存重采样种子。"""
        names = ('patches', 'patch_masks', 'edge_bands', 'edge_masks', 'edge_valid_flags',
                 'source_img_ids', 'source_coords', 'content_feats', 'edge_feats', 'pred_positions')
        return {
            'pool_size': self.pool_size, 'grid_size': self.grid_size, 'seed': self.seed,
            'sampled_tuples': self.sampled_tuples,
            'content_encoder_state': ({k: v.detach().cpu().clone() for k, v in self._content_snapshot.state_dict().items()}
                                      if self._content_snapshot is not None else None),
            'edge_encoder_state': ({k: v.detach().cpu().clone() for k, v in self._edge_snapshot.state_dict().items()}
                                   if self._edge_snapshot is not None else None),
            'tensors': {name: getattr(self, name).detach().cpu().clone()
                        for name in names if getattr(self, name) is not None},
        }

    @classmethod
    def from_state_dict(cls, state, device=torch.device('cpu')):
        pool = cls(state['pool_size'], state['grid_size'], device=device, seed=state['seed'])
        pool.sampled_tuples = state.get('sampled_tuples', [])
        for name, value in state['tensors'].items():
            setattr(pool, name, value.to(device))
        if state.get('content_encoder_state') is not None and state.get('edge_encoder_state') is not None:
            from pintu_v2.models.content_encoder import ContentEncoder
            from pintu_v2.models.edge_encoder import EdgeEncoder
            pool._content_snapshot = ContentEncoder(pool.content_feats.shape[-1]).to(device).eval().requires_grad_(False)
            pool._edge_snapshot = EdgeEncoder(pool.edge_feats.shape[-1]).to(device).eval().requires_grad_(False)
            pool._content_snapshot.load_state_dict(state['content_encoder_state'])
            pool._edge_snapshot.load_state_dict(state['edge_encoder_state'])
        for name in ('patches', 'content_feats', 'edge_feats', 'pred_positions', 'source_img_ids'):
            value = getattr(pool, name)
            if value is None or value.shape[0] != pool.pool_size:
                raise ValueError(f'检查点候选池缺失或损坏: {name}')
        return pool


class BatchInventory:
    """
    批次共享库存管理器 (Batch-Shared Inventory):
    - 整个批次共同消耗同一份大小为 [K] 的候选库存 (consumed_inventory=True 模式)。
    - 保留每个样本的同源排除掩码 excluded_mask (B, K)，禁止挑选来自自身原图的 Patch。
    - 在选块分配时进行冲突仲裁，检索前严密检查是否有可用合法候选。
    - 在不消耗模式下，追踪连续可微的软概率累加分布 (soft_usage)。
    """
    def __init__(
        self,
        batch_size: int,
        pool_size: int,
        source_img_ids: torch.Tensor,
        seed_img_ids: torch.Tensor,
        consumed_inventory: bool = True,
        allow_self_patches: bool = True,
        device: torch.device = torch.device("cpu"),
        inventory_scope: str = "batch"
    ):
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.consumed_inventory = consumed_inventory
        self.allow_self_patches = allow_self_patches
        self.device = device
        if inventory_scope not in ('batch', 'image'):
            raise ValueError('inventory_scope 必须为 batch 或 image')
        self.inventory_scope = inventory_scope

        # 1. 批次共享消耗状态向量 [K]: True 表示已被当前批次中某个样本消耗
        shape = (pool_size,) if inventory_scope == 'batch' else (batch_size, pool_size)
        self.consumed_mask = torch.zeros(shape, dtype=torch.bool, device=device)

        # 2. 每个样本独立的同图排除矩阵 (B, K): True 表示该候选来自该样本的同一原图 (当 allow_self_patches=False 时生效)
        if self.allow_self_patches:
            self.excluded_mask = torch.zeros((batch_size, pool_size), dtype=torch.bool, device=device)
        else:
            self.excluded_mask = (seed_img_ids.unsqueeze(1) == source_img_ids.unsqueeze(0)).to(device)

        # 3. 统计使用情况 (用于不消耗模式下的惩罚)
        self.hard_usage_counts = torch.zeros((batch_size, pool_size), dtype=torch.float32, device=device)
        self.soft_usage = torch.zeros((batch_size, pool_size), dtype=torch.float32, device=device)
        self.soft_squared_sum = torch.zeros(batch_size, device=device)
        self.soft_steps = 0

    def get_valid_mask(self) -> torch.Tensor:
        """
        获取当前批次每个样本的所有可用候选块掩码 (B, K)。
        True 表示该候选块对该样本合法可用。
        """
        if self.consumed_inventory:
            # 批次共享已消耗 + 样本独立同源排除
            shared_avail = ~self.consumed_mask
            if self.inventory_scope == 'batch':
                shared_avail = shared_avail.unsqueeze(0)
            return shared_avail & (~self.excluded_mask)      # (B, K)
        else:
            return ~self.excluded_mask

    def check_availability(self):
        """
        检查每个样本是否仍有合法候选块。若耗尽则抛出异常，防止无声失效。
        """
        valid_counts = self.get_valid_mask().sum(dim=-1) # (B,)
        exhausted = (valid_counts == 0).nonzero(as_tuple=True)[0]
        if len(exhausted) > 0:
            raise RuntimeError(
                f"候选池库存耗尽！批次内样本 {exhausted.tolist()} 均已无可用的合法候选块。"
                f"(当前池大小: {self.pool_size}, 已消耗数: {self.consumed_mask.sum().item()})"
            )

    def consume(self, selected_indices: torch.Tensor):
        """
        标记选中候选块已消耗。
        selected_indices: (B,)
        """
        b_indices = torch.arange(self.batch_size, device=self.device)
        if not self.get_valid_mask()[b_indices, selected_indices].all():
            raise RuntimeError('不能消耗已使用或同源排除的候选块')
        if self.consumed_inventory and self.inventory_scope == 'batch' and selected_indices.unique().numel() != self.batch_size:
            raise RuntimeError('批次内存在重复分配的候选块')
        if self.consumed_inventory:
            # 整批共同消耗
            if self.inventory_scope == 'batch':
                self.consumed_mask[selected_indices] = True
            else:
                self.consumed_mask[b_indices, selected_indices] = True
        self.hard_usage_counts[b_indices, selected_indices] += 1.0

    def record_soft_usage(self, soft_weights: torch.Tensor):
        """
        记录可微的软概率分布以计算重复使用惩罚。
        soft_weights: (B, K)
        """
        self.soft_usage = self.soft_usage + soft_weights
        self.soft_squared_sum = self.soft_squared_sum + soft_weights.square().sum(dim=-1)
        self.soft_steps += 1

    def get_repetition_loss(self) -> torch.Tensor:
        """
        不消耗模式下的连续可微重复惩罚:
        计算各步软分布的平均成对碰撞概率；硬重复约束另由检索分数惩罚提供。
        """
        # 各次选择分布的成对碰撞概率。大池下 soft_usage 通常小于 1，
        # 原来的 hinge 会长期恒为零，无法约束硬选择重复。
        if self.soft_steps < 2:
            return self.soft_usage.sum() * 0.0
        collisions = self.soft_usage.square().sum(dim=-1) - self.soft_squared_sum
        return (collisions / (self.soft_steps * (self.soft_steps - 1))).mean().clamp_min(0.0)
