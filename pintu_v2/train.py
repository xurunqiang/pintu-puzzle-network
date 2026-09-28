"""
“拼图”网络第二版：两阶段自监督训练主程序 (train.py)
阶段 1 (Warmup 1-20 轮): 预热内容编码器、边缘编码器、5分类均衡方向分类头、位置预测头与全图 VICReg
阶段 2 (Full 21-100 轮): 启用 8,192 候选池，FIFO 队列扩展，双拼图视图 VICReg 与动量教师对齐
"""

import os
import sys
import time
import argparse
import copy
import tempfile
from dataclasses import asdict
from typing import Dict, Any, Tuple

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from pintu_v2.config import ModelConfig, PoolConfig, TrainConfig
from pintu_v2.data.cifar_dataset import get_cifar_dataloaders
from pintu_v2.data.patch_pool import PatchPool
from pintu_v2.models.pintu_model import PintuModel
from pintu_v2.losses.composite_loss import CompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="训练拼图网络第二版")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/pintu_v2", help="模型检查点保存目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="阶段 1 预热轮数")
    parser.add_argument("--total_epochs", type=int, default=100, help="总轮数 (阶段 1 + 阶段 2)")
    parser.add_argument("--grid_size", type=int, default=5, choices=[3, 5, 7], help="拼图网格大小")
    parser.add_argument("--pool_size", type=int, default=8192, help="候选块池大小")
    parser.add_argument("--cross_pool_only", action='store_true', help="消融：不保证原图碎片入池，沿用纯跨图候选")
    parser.add_argument("--consumed_inventory", action="store_true", default=True, help="是否使用批次共享库存消耗模式")
    parser.add_argument("--no_consumed_inventory", dest="consumed_inventory", action="store_false", help="使用不消耗库存模式 (带重复惩罚)")
    parser.add_argument("--inventory_scope", choices=['batch', 'image'], default='batch', help="消耗库存范围：批次共享或每张图独立")
    parser.add_argument("--feature_refresh_steps", type=int, default=PoolConfig().feature_refresh_steps)
    parser.add_argument("--momentum_decay", type=float, default=PoolConfig().momentum_decay)
    parser.add_argument("--resume", type=str, default=None, help="恢复本副本生成的完整训练检查点")
    parser.add_argument("--allow_self_patches", action="store_true", default=True, help="允许选择池中已有的同源候选；不保证包含原图全部块")
    parser.add_argument("--exclude_self_patches", dest="allow_self_patches", action="store_false", help="禁止挑选自身原图候选块 (严格同源排除)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--quick_test", action="store_true", help="冒烟测试模式 (极少轮数与极小池)")
    return parser.parse_args()


def save_checkpoint_atomic(state, path):
    """写完临时文件后再替换目标，避免中断破坏上一份可续训检查点。"""
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path) or '.', suffix='.tmp', delete=False) as handle:
        temporary = handle.name
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def update_momentum_model(model: PintuModel, momentum_model: PintuModel, decay: float = 0.99):
    """
    动量 EMA 参数更新: theta_m = decay * theta_m + (1 - decay) * theta
    """
    for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
        param_k.data.mul_(decay).add_(param_q.data, alpha=1.0 - decay)


def extract_batch_training_pairs(model: PintuModel, imgs: torch.Tensor, device: torch.device):
    """
    从一批完整图像中提取均衡的 5 分类方向样本 (上、右、下、左、非相邻) 及对应真实接缝边带。
    """
    B = imgs.shape[0]
    g = model.grid_size
    denom = max(1, g - 1)

    patches, masks = model.slicer.slice_image(imgs) # (B, g, g, 3, P, P), (B, g, g, 1, P, P)

    # 1. 位置预测样本
    sample_r = torch.randint(0, g, (B,), device=device)
    sample_c = torch.randint(0, g, (B,), device=device)
    b_idx = torch.arange(B, device=device)
    
    sel_patches = patches[b_idx, sample_r, sample_c] # (B, 3, P, P)
    _, pred_pos = model.content_encoder(sel_patches)
    target_pos = torch.stack([sample_r.float() / denom, sample_c.float() / denom], dim=1) # (B, 2)

    # 2. 相对方向 5 分类均衡采样 (0: 上, 1: 右, 2: 下, 3: 左, 4: 非相邻)
    direction_labels = ((torch.arange(B) + torch.randint(0, 5, ())) % 5).tolist()
    dir_targets = torch.tensor(direction_labels, device=device)

    patch_a_list = []
    patch_b_list = []
    mask_a_list = []
    mask_b_list = []
    side_a_list = []
    side_b_list = []
    is_adjacent_list = []

    for b in range(B):
        target_dir = direction_labels[b]
        if target_dir == 0: # B 在 A 上方 (r_B = r_A - 1, c_B = c_A)
            ra = torch.randint(1, g, (1,)).item()
            ca = torch.randint(0, g, (1,)).item()
            rb, cb = ra - 1, ca
            side_a, side_b = 'top', 'bottom'
            is_adj = True
            b_src = b
        elif target_dir == 1: # B 在 A 右侧 (r_B = r_A, c_B = c_A + 1)
            ra = torch.randint(0, g, (1,)).item()
            ca = torch.randint(0, g - 1, (1,)).item()
            rb, cb = ra, ca + 1
            side_a, side_b = 'right', 'left'
            is_adj = True
            b_src = b
        elif target_dir == 2: # B 在 A 下方 (r_B = r_A + 1, c_B = c_A)
            ra = torch.randint(0, g - 1, (1,)).item()
            ca = torch.randint(0, g, (1,)).item()
            rb, cb = ra + 1, ca
            side_a, side_b = 'bottom', 'top'
            is_adj = True
            b_src = b
        elif target_dir == 3: # B 在 A 左侧 (r_B = r_A, c_B = c_A - 1)
            ra = torch.randint(0, g, (1,)).item()
            ca = torch.randint(1, g, (1,)).item()
            rb, cb = ra, ca - 1
            side_a, side_b = 'left', 'right'
            is_adj = True
            b_src = b
        else: # 非相邻必须由同图真实坐标定义；跨图块仍可能在语义上适合相邻。
            ra = torch.randint(0, g, (1,)).item()
            ca = torch.randint(0, g, (1,)).item()
            candidates = [(r, c) for r in range(g) for c in range(g)
                          if abs(r - ra) + abs(c - ca) > 1]
            rb, cb = candidates[torch.randint(len(candidates), (1,)).item()]
            side_a, side_b = 'right', 'left'
            is_adj = False
            b_src = b

        patch_a_list.append(patches[b, ra, ca])
        patch_b_list.append(patches[b_src, rb, cb])
        mask_a_list.append(masks[b, ra, ca])
        mask_b_list.append(masks[b_src, rb, cb])
        side_a_list.append(side_a)
        side_b_list.append(side_b)
        is_adjacent_list.append(is_adj)

    p_a = torch.stack(patch_a_list, dim=0) # (B, 3, P, P)
    p_b = torch.stack(patch_b_list, dim=0)
    m_a = torch.stack(mask_a_list, dim=0)
    m_b = torch.stack(mask_b_list, dim=0)

    feat_a, _ = model.content_encoder(p_a)
    feat_b, _ = model.content_encoder(p_b)
    dir_logits = model.content_encoder.predict_direction(feat_a, feat_b) # (B, 5)

    # 3. 提取相邻接缝边带特征 (仅对 0, 1, 2, 3 相邻对计算物理边缘对比)
    bands_a, b_masks_a = model.slicer.extract_edge_bands(p_a, m_a)
    bands_b, b_masks_b = model.slicer.extract_edge_bands(p_b, m_b)

    adjacent = [b for b in range(B) if is_adjacent_list[b]]
    if adjacent:
        # 合并两侧边带，一次编码；保持正样本配对顺序。
        inputs = torch.stack([bands_a[side_a_list[b]][b] for b in adjacent] +
                             [bands_b[side_b_list[b]][b] for b in adjacent])
        masks = torch.stack([b_masks_a[side_a_list[b]][b] for b in adjacent] +
                            [b_masks_b[side_b_list[b]][b] for b in adjacent])
        edges, valid = model.edge_encoder(inputs, masks)
        ea, eb = edges.chunk(2)
        va, vb = valid.chunk(2)
        edge_pairs = (ea[va & vb], eb[va & vb])
    else:
        edge_pairs = None

    return {
        "pred_pos": pred_pos,
        "target_pos": target_pos,
        "dir_logits": dir_logits,
        "dir_targets": dir_targets,
        "edge_pairs": edge_pairs
    }


@torch.no_grad()
def evaluate_validation_loss(
    model: PintuModel,
    momentum_model: PintuModel,
    loss_fn: CompositeLoss,
    val_loader,
    phase: str,
    pool: PatchPool,
    device: torch.device,
    tau: float = 0.2,
    max_batches: int = 15,
    consumed_inventory: bool = True,
    allow_self_patches: bool = True,
    validation_seed: int = 42,
    inventory_scope: str = 'batch'
) -> float:
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(validation_seed)
        try:
            return _evaluate_validation_loss(model, momentum_model, loss_fn, val_loader,
                                             phase, pool, device, tau, max_batches,
                                             consumed_inventory, allow_self_patches, inventory_scope)
        finally:
            model.train(was_training)


@torch.no_grad()
def _evaluate_validation_loss(model, momentum_model, loss_fn, val_loader, phase, pool,
                              device, tau, max_batches, consumed_inventory,
                              allow_self_patches: bool = True, inventory_scope: str = 'batch'):
    """
    在真实的验证集 (val_loader) 上计算平均验证损失。
    """
    model.eval()
    total_val_loss = 0.0
    val_count = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        raw_imgs = batch["raw_img"].to(device)
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        img_ids = batch["img_id"].to(device)
        if raw_imgs.shape[0] < 2:
            continue  # VICReg 的样本方差至少需要两个样本。

        z_v1 = model.encode_full_image(view1)
        z_v2 = model.encode_full_image(view2)
        aux_samples = extract_batch_training_pairs(model, raw_imgs, device)

        z_puzzle1 = None
        z_puzzle2 = None
        z_teacher = None
        rep_loss = None

        if phase == "full":
            puzzle_res1 = model.assemble_puzzle(view1, img_ids, pool, tau=tau,
                                                consumed_inventory=consumed_inventory,
                                                record_trajectory=False,
                                                allow_self_patches=allow_self_patches, inventory_scope=inventory_scope)
            puzzle_res2 = model.assemble_puzzle(view2, img_ids, pool, tau=tau,
                                                consumed_inventory=consumed_inventory,
                                                record_trajectory=False,
                                                allow_self_patches=allow_self_patches, inventory_scope=inventory_scope)
            z_puzzle1 = puzzle_res1["z_puzzle"]
            z_puzzle2 = puzzle_res2["z_puzzle"]
            z_teacher = momentum_model.encode_full_image(view2)
            rep_loss = (puzzle_res1["repetition_loss"] + puzzle_res2["repetition_loss"]) / 2

        loss, _ = loss_fn(
            phase=phase,
            z_v1=z_v1,
            z_v2=z_v2,
            edge_pairs=aux_samples["edge_pairs"],
            dir_logits=aux_samples["dir_logits"],
            dir_targets=aux_samples["dir_targets"],
            pred_pos=aux_samples["pred_pos"],
            target_pos=aux_samples["target_pos"],
            z_puzzle1=z_puzzle1,
            z_puzzle2=z_puzzle2,
            z_teacher=z_teacher,
            repetition_loss=rep_loss
        )
        total_val_loss += loss.item() * raw_imgs.shape[0]
        val_count += raw_imgs.shape[0]

    if not val_count:
        raise ValueError('验证集没有可计算 VICReg 的批次')
    return total_val_loss / val_count


def train_pintu_v2(args):
    if not args.cross_pool_only and not args.allow_self_patches:
        raise ValueError('默认保证原图碎片可选；--exclude_self_patches 需要同时指定 --cross_pool_only')
    if args.feature_refresh_steps < 1 or not 0 <= args.momentum_decay < 1:
        raise ValueError('刷新间隔必须为正整数，momentum_decay 必须在 [0, 1)')
    if not args.quick_test and not 0 <= args.warmup_epochs < args.total_epochs:
        raise ValueError('需要 0 <= warmup_epochs < total_epochs，保证有完整拼图训练阶段')
    if args.batch_size < 2:
        raise ValueError('自监督 VICReg 训练需要 batch_size >= 2')
    if args.quick_test:
        args.batch_size = min(args.batch_size, 4)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resume.get('implementation_version') != 4:
            raise ValueError('续训需要本优化副本生成的检查点；旧版权重请用于单独评估')
        for name in ('grid_size', 'batch_size', 'seed', 'consumed_inventory', 'inventory_scope',
                     'allow_self_patches', 'feature_refresh_steps', 'momentum_decay',
                     'total_epochs', 'warmup_epochs', 'quick_test', 'lr', 'weight_decay', 'cross_pool_only'):
            if resume['args'].get(name) != getattr(args, name):
                raise ValueError(f'续训参数 {name} 必须与检查点一致')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"=== 启动“拼图”网络第二版训练 ===")
    print(f"设备: {device}, 网格: {args.grid_size}x{args.grid_size}, 批次大小: {args.batch_size}")
    print(f"库存消耗: {args.consumed_inventory}, 库存范围: {args.inventory_scope}, 允许同源候选: {args.allow_self_patches}")
    total_epochs = 2 if args.quick_test else args.total_epochs
    warmup_epochs = 1 if args.quick_test else args.warmup_epochs
    print(f"预热轮数: {warmup_epochs}, 总轮数: {total_epochs}")

    # 1. 数据准备
    train_loader, val_loader, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2 if args.quick_test else 4,
        seed=args.seed
    )
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset

    pool_size = max(128, (args.batch_size + 1) * (args.grid_size ** 2 - 1)) if args.quick_test else args.pool_size
    required_pool = (args.batch_size if args.inventory_scope == 'batch' else 1) * (args.grid_size ** 2 - 1)
    if not args.cross_pool_only:
        required_pool = args.batch_size * (args.grid_size ** 2 - 1)
        if pool_size < required_pool:
            raise ValueError(f'包含本批全部原图碎片至少需要 {required_pool} 个候选槽位')
    if args.consumed_inventory and pool_size < required_pool:
        raise ValueError(f'{args.inventory_scope} 库存模式至少需要 {required_pool} 个候选')

    # 2. 候选池构建 (严格从 train_subset 建池，并断言排除验证集)
    if resume is None:
        print(f"正在从 train_subset 构建 {pool_size} 大小的候选池...")
        patch_pool = PatchPool(pool_size=pool_size, grid_size=args.grid_size, device=device, seed=args.seed)
        patch_pool.build_pool(train_subset)
    else:
        patch_pool = PatchPool.from_state_dict(resume['pool_state'], device)
        if patch_pool.pool_size != pool_size or patch_pool.grid_size != args.grid_size:
            raise ValueError('续训候选池大小和网格必须与检查点一致')
        if not set(patch_pool.source_img_ids.cpu().tolist()).issubset(set(train_subset.indices)):
            raise ValueError('续训候选池不属于当前训练划分')

    # 验证候选池与验证集无交集 (无泄漏断言)
    val_real_ids = set(val_subset.indices)
    pool_source_ids = set(patch_pool.source_img_ids.tolist())
    assert pool_source_ids.isdisjoint(val_real_ids), "致命错误: 候选池混入了验证集图像！"
    print(f"✓ 候选池无泄漏验证通过! 来源图片数: {len(pool_source_ids)}, 与验证集完全无交集。")

    # 3. 模型与动量分支构建
    model_cfg = ModelConfig(grid_size=args.grid_size, include_source_patches=not args.cross_pool_only)
    model = PintuModel(cfg=model_cfg).to(device)

    momentum_model = copy.deepcopy(model).to(device)
    for p in momentum_model.parameters():
        p.requires_grad = False
    momentum_model.eval()

    # 4. 优化器与损失函数
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs
    )
    loss_fn = CompositeLoss(cfg=train_cfg)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_warmup = min(train_cfg.lr_warmup_epochs, max(1, total_epochs - 1))
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, total_iters=lr_warmup),
        CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - lr_warmup), eta_min=1e-6),
    ], milestones=[lr_warmup])

    # 初始刷新池中特征
    if resume is None:
        patch_pool.refresh_features(model.content_encoder, model.edge_encoder)

    best_warmup_val_loss = float('inf')
    best_full_val_loss = float('inf')
    global_step = 0

    start_epoch = 1
    if resume is not None:
        if resume['model_config'] != asdict(model_cfg):
            raise ValueError('续训的 ModelConfig 必须与检查点一致')
        current_training = {k: v for k, v in asdict(train_cfg).items() if k not in ('data_dir', 'save_dir')}
        saved_training = {k: v for k, v in resume['train_config'].items() if k not in ('data_dir', 'save_dir')}
        if saved_training != current_training:
            raise ValueError('续训的 TrainConfig 必须与检查点一致')
        model.load_state_dict(resume['model_state'])
        momentum_model.load_state_dict(resume['momentum_model_state'])
        optimizer.load_state_dict(resume['optimizer_state'])
        scheduler.load_state_dict(resume['scheduler_state'])
        global_step = resume['global_step']
        best_warmup_val_loss = resume['best_warmup_val_loss']
        best_full_val_loss = resume['best_full_val_loss']
        start_epoch = resume['epoch'] + 1
        torch.random.set_rng_state(resume['rng_state'])
        if device.type == 'cuda' and resume.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all(resume['cuda_rng_states'])
        print(f'已恢复检查点，将从第 {start_epoch} 轮继续')
        del resume

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        phase = "warmup" if epoch <= warmup_epochs else "full"

        # 阶段切换即时刷新候选池特征 (Item 补齐)
        if epoch == warmup_epochs + 1:
            print(f"\n>>> 进入完整拼图阶段 (Phase 2)，立即使用动量模型全面刷新候选池特征...")
            patch_pool.refresh_features(momentum_model.content_encoder, momentum_model.edge_encoder)

        model.train()

        # 温度退火
        if phase == "warmup":
            tau = 1.0
        else:
            progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            tau = model_cfg.tau_start - progress * (model_cfg.tau_start - model_cfg.tau_end)
            tau = max(tau, model_cfg.tau_end)

        running_loss = 0.0
        print(f"\n--- Epoch [{epoch}/{total_epochs}] | 阶段: {phase.upper()} | 学习率: {scheduler.get_last_lr()[0]:.6f} | Tau: {tau:.3f} ---")

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            raw_imgs = batch["raw_img"].to(device)
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)
            img_ids = batch["img_id"].to(device)

            # 1. 全图基础表示
            z_v1 = model.encode_full_image(view1)
            z_v2 = model.encode_full_image(view2)

            # 2. 5 分类均衡方位与边缘对比样本
            aux_samples = extract_batch_training_pairs(model, raw_imgs, device)

            # 3. 完整拼图阶段: 双拼图视图生成与对齐
            z_puzzle1 = None
            z_puzzle2 = None
            z_teacher = None
            rep_loss = None

            if phase == "full":
                # 定期刷新候选池特征
                if global_step % args.feature_refresh_steps == 0:
                    patch_pool.refresh_features(momentum_model.content_encoder, momentum_model.edge_encoder)

                # 视图 1 组装拼图
                puzzle_res1 = model.assemble_puzzle(
                    seed_imgs=view1,
                    seed_img_ids=img_ids,
                    pool=patch_pool,
                    tau=tau,
                    consumed_inventory=args.consumed_inventory,
                    record_trajectory=False,
                    allocation_offset=global_step,
                    allow_self_patches=args.allow_self_patches,
                    inventory_scope=args.inventory_scope
                )
                z_puzzle1 = puzzle_res1["z_puzzle"]
                rep_loss = puzzle_res1["repetition_loss"]

                # 视图 2 组装拼图 (用于双拼图视图一致性学习)
                puzzle_res2 = model.assemble_puzzle(
                    seed_imgs=view2,
                    seed_img_ids=img_ids,
                    pool=patch_pool,
                    tau=tau,
                    consumed_inventory=args.consumed_inventory,
                    record_trajectory=False,
                    allocation_offset=global_step,
                    allow_self_patches=args.allow_self_patches,
                    inventory_scope=args.inventory_scope
                )
                z_puzzle2 = puzzle_res2["z_puzzle"]
                rep_loss = (rep_loss + puzzle_res2['repetition_loss']) / 2

                # 动量全图教师表示
                with torch.no_grad():
                    z_teacher = momentum_model.encode_full_image(view2)

            # 4. 计算综合损失
            loss, loss_dict = loss_fn(
                phase=phase,
                z_v1=z_v1,
                z_v2=z_v2,
                edge_pairs=aux_samples["edge_pairs"],
                dir_logits=aux_samples["dir_logits"],
                dir_targets=aux_samples["dir_targets"],
                pred_pos=aux_samples["pred_pos"],
                target_pos=aux_samples["target_pos"],
                z_puzzle1=z_puzzle1,
                z_puzzle2=z_puzzle2,
                z_teacher=z_teacher,
                repetition_loss=rep_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            update_momentum_model(model, momentum_model, decay=args.momentum_decay)
            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0 or (args.quick_test and batch_idx == 0):
                log_str = f"Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} "
                if "loss_vicreg_puzzle" in loss_dict:
                    log_str += f"| PuzVIC: {loss_dict['loss_vicreg_puzzle']:.3f} Align: {loss_dict['loss_align']:.3f} "
                    log_str += f"Repeat: {puzzle_res1['hard_repeat_rate'].mean().item():.1%} "
                log_str += f"| Edge: {loss_dict.get('loss_edge', 0.0):.3f} Dir: {loss_dict.get('loss_dir', 0.0):.3f}"
                print(log_str)

            if args.quick_test and batch_idx >= 2:
                break

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_train_loss = running_loss / (batch_idx + 1)

        # 5. 在真实的验证集上评估验证损失并分阶段保存最佳模型
        val_batches = 3 if args.quick_test else 20
        val_loss = evaluate_validation_loss(
            model=model,
            momentum_model=momentum_model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            phase=phase,
            pool=patch_pool,
            device=device,
            tau=tau,
            max_batches=val_batches,
            consumed_inventory=args.consumed_inventory,
            allow_self_patches=args.allow_self_patches,
            validation_seed=args.seed,
            inventory_scope=args.inventory_scope
        )
        print(f"Epoch [{epoch}] 耗时: {epoch_time:.2f}s | 训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")

        # 打包保存状态
        ckpt_meta = {
            "implementation_version": 4,
            "best_warmup_val_loss": min(best_warmup_val_loss, val_loss) if phase == 'warmup' else best_warmup_val_loss,
            "best_full_val_loss": min(best_full_val_loss, val_loss) if phase == 'full' else best_full_val_loss,
            "rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            "epoch": epoch,
            "phase": phase,
            "grid_size": args.grid_size,
            "pool_size": pool_size,
            "pool_tuples": patch_pool.sampled_tuples,
            "seed": args.seed,
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "pool_state": patch_pool.state_dict(),
            "model_state": model.state_dict(),
            "momentum_model_state": momentum_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "args": vars(args)
        }

        save_checkpoint_atomic(ckpt_meta, os.path.join(args.save_dir, "latest_checkpoint.pth"))

        if phase == "warmup" and val_loss < best_warmup_val_loss:
            best_warmup_val_loss = val_loss
            save_checkpoint_atomic(ckpt_meta, os.path.join(args.save_dir, "best_warmup_checkpoint.pth"))
            print(f"★ 最佳预热检查点已更新并保存至 best_warmup_checkpoint.pth (Val Loss: {val_loss:.4f})")

        if phase == "full" and val_loss < best_full_val_loss:
            best_full_val_loss = val_loss
            save_checkpoint_atomic(ckpt_meta, os.path.join(args.save_dir, "best_full_checkpoint.pth"))
            save_checkpoint_atomic(ckpt_meta, os.path.join(args.save_dir, "best_checkpoint.pth"))
            print(f"★ 最佳完整拼图检查点已更新并保存至 best_checkpoint.pth (Val Loss: {val_loss:.4f})")

    print("\n训练流程已全部顺利结束！")


if __name__ == "__main__":
    cli_args = parse_args()
    train_pintu_v2(cli_args)
