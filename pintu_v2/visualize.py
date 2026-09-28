"""
“拼图”网络第二版：组装过程与拼图效果可视化 (visualize.py)
输出:
1. 种子中心块 vs 原图 vs 拼图结果对比
2. 24 步队列扩展拼装动态序列过程图
3. 块与块之间的接缝过渡热力图
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from pintu_v2.config import ModelConfig
from pintu_v2.data.cifar_dataset import get_cifar_dataloaders
from pintu_v2.data.patch_pool import PatchPool
from pintu_v2.data.patch_slicer import PatchSlicer
from pintu_v2.checkpoint import (
    load_model_checkpoint, restore_pool, checkpoint_seed,
    checkpoint_inventory_mode, checkpoint_allow_self_patches, checkpoint_inventory_scope
)


def parse_args():
    parser = argparse.ArgumentParser(description="拼图组装过程可视化")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pintu_v2/best_checkpoint.pth", help="模型权重")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="可视化图像输出目录")
    parser.add_argument("--num_samples", type=int, default=3, help="展示的种子图像数量")
    parser.add_argument("--batch_size", type=int, default=None, help="实际组装批次大小，默认沿用检查点，与展示数量独立")
    parser.add_argument("--inventory_scope", choices=['batch', 'image'], default=None, help="默认沿用检查点库存范围")
    parser.add_argument("--allow_self_patches", dest="allow_self_patches", action="store_true", default=None, help="允许挑选自身原图候选块 (拼回原图)")
    parser.add_argument("--exclude_self_patches", dest="allow_self_patches", action="store_false", help="禁止挑选自身原图候选块 (严格同源排除)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    return parser.parse_args()


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """
    (3, H, W) in [0, 1] -> (H, W, 3) in [0, 1]
    """
    t = torch.clamp(t, 0.0, 1.0)
    return t.permute(1, 2, 0).cpu().numpy()


def visualize_puzzle_results(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== 运行拼图网络第二版组装与轨迹可视化 ===")

    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    grid_size = model.grid_size
    batch_size = args.batch_size or checkpoint.get('args', {}).get('batch_size', 64)
    if not 1 <= args.num_samples <= batch_size:
        raise ValueError('需要 1 <= num_samples <= batch_size')
    scope = args.inventory_scope or checkpoint_inventory_scope(checkpoint)
    print(f'组装批次: {batch_size}，展示数量: {args.num_samples}，库存范围: {scope}')
    train_loader, _, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir, batch_size=batch_size, num_workers=2,
        seed=checkpoint_seed(checkpoint), for_evaluation=True,
    )
    pool = restore_pool(checkpoint, model, train_loader.dataset, device)

    # 3. 抽取测试样本进行拼图
    test_batch = next(iter(test_loader))
    seed_imgs = test_batch["raw_img"].to(device)
    seed_ids = test_batch["img_id"].to(device)
    labels = test_batch["label"][:args.num_samples]

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    allow_self = args.allow_self_patches if args.allow_self_patches is not None else checkpoint_allow_self_patches(checkpoint)
    with torch.no_grad():
        puzzle_out = model.assemble_puzzle(
            seed_imgs=seed_imgs,
            seed_img_ids=seed_ids,
            pool=pool,
            tau=0.2,
            consumed_inventory=checkpoint_inventory_mode(checkpoint),
            record_trajectory=True,
            allow_self_patches=allow_self,
            inventory_scope=scope
        )

    assembled_imgs = puzzle_out["puzzle_img"] # (B, 3, 35, 35)
    slicer = PatchSlicer(grid_size=grid_size)
    seed_patches, _ = slicer.slice_image(seed_imgs)
    cr, cc = slicer.center_coord

    # 4. 生成对比总览图
    fig, axes = plt.subplots(args.num_samples, 3, figsize=(10, 3.2 * args.num_samples))
    if args.num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(args.num_samples):
        cls_name = classes[labels[i]]
        # 种子中心块
        center_p = seed_patches[i, cr, cc] # (3, 7, 7)
        axes[i, 0].imshow(tensor_to_img(center_p))
        axes[i, 0].set_title(f"Seed Center ({cls_name})", fontsize=11)
        axes[i, 0].axis('off')

        # 原始图像 (32x32)
        axes[i, 1].imshow(tensor_to_img(seed_imgs[i]))
        axes[i, 1].set_title("Original Ground Truth", fontsize=11)
        axes[i, 1].axis('off')

        # 组装拼图结果 (35x35)
        axes[i, 2].imshow(tensor_to_img(assembled_imgs[i]))
        axes[i, 2].set_title("Assembled Cross-Pool Puzzle", fontsize=11)
        axes[i, 2].axis('off')

    plt.tight_layout()
    overview_path = os.path.join(args.output_dir, "puzzle_overview_comparison.png")
    plt.savefig(overview_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"★ 拼图对比总览已保存至: {overview_path}")

    # 5. 生成第 1 个样本的 24 步队列扩展全流程图
    sample_idx = 0
    traj = puzzle_out["trajectory"] # 24 步
    g = grid_size
    P = slicer.patch_size

    # 展示步骤采样: 选取步数 0, 2, 5, 8, 12, 16, 20, 23
    key_steps = np.linspace(0, len(traj) - 1, 8, dtype=int).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    # 逐步复原拼图状态
    curr_canvas = np.zeros((g * P, g * P, 3), dtype=np.float32)
    # 先填入中心块
    c_p_np = tensor_to_img(seed_patches[sample_idx, cr, cc])
    curr_canvas[cr*P:(cr+1)*P, cc*P:(cc+1)*P] = c_p_np

    step_snapshots = {}
    for step_info in traj:
        step_num = step_info["step"]
        tr, tc = step_info["target_coord"]
        sel_idx = step_info["selected_idx"][sample_idx].item()
        
        # 填入挑选块
        chosen_p = tensor_to_img(pool.patches[sel_idx])
        curr_canvas[tr*P:(tr+1)*P, tc*P:(tc+1)*P] = chosen_p

        if step_num in key_steps:
            step_snapshots[step_num] = curr_canvas.copy()

    for ax_idx, k_step in enumerate(key_steps):
        snap = step_snapshots.get(k_step, curr_canvas)
        axes[ax_idx].imshow(snap)
        axes[ax_idx].set_title(f"Step {k_step + 1} / {len(traj)}", fontsize=11)
        axes[ax_idx].axis('off')

    plt.suptitle(f"Progressive Queue Expansion & Assembly Trajectory ({g}x{g} Grid)", fontsize=14, y=0.98)
    plt.tight_layout()
    prog_path = os.path.join(args.output_dir, "puzzle_progression_trajectory.png")
    plt.savefig(prog_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"★ 拼图逐步扩展轨迹图已保存至: {prog_path}")

    # 接缝热力图：仅统计两侧真实有效像素，灰色表示无证据或非接缝位置。
    pieces = puzzle_out['assembled_patches'][:args.num_samples].detach()
    masks = puzzle_out['assembled_masks'][:args.num_samples]
    heatmaps = np.full((args.num_samples, 2 * g - 1, 2 * g - 1), np.nan)
    with torch.no_grad():
        for r in range(g):
            for c in range(g):
                bands_a, masks_a = slicer.extract_edge_bands(pieces[:, r, c], masks[:, r, c])
                for dr, dc, side_a, side_b in [(0, 1, 'right', 'left'), (1, 0, 'bottom', 'top')]:
                    if r + dr >= g or c + dc >= g:
                        continue
                    bands_b, masks_b = slicer.extract_edge_bands(pieces[:, r + dr, c + dc], masks[:, r + dr, c + dc])
                    color, gradient = slicer.compute_raw_seam_difference(
                        bands_a[side_a], bands_b[side_b], masks_a[side_a], masks_b[side_b])
                    valid = (masks_a[side_a][:, 0, 0] * masks_b[side_b][:, 0, 0]).sum(-1) > 0
                    value = (color + gradient).masked_fill(~valid, float('nan')).cpu().numpy()
                    heatmaps[:, 2 * r + dr, 2 * c + dc] = value
    fig, axes = plt.subplots(1, args.num_samples, figsize=(4 * args.num_samples, 4), squeeze=False,
                             constrained_layout=True)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#dddddd')
    maximum = float(np.nanmax(heatmaps)) if np.isfinite(heatmaps).any() else 1.0
    for i, ax in enumerate(axes[0]):
        plot = ax.imshow(heatmaps[i], cmap=cmap, vmin=0, vmax=max(maximum, 1e-6))
        ax.set_title(f'Seam error: sample {i + 1}')
        ax.axis('off')
    fig.colorbar(plot, ax=list(axes[0]), label='Color L1 + normal gradient error')
    seam_path = os.path.join(args.output_dir, 'puzzle_seam_heatmap.png')
    fig.savefig(seam_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'★ 接缝热力图已保存至: {seam_path}')


if __name__ == "__main__":
    cli_args = parse_args()
    visualize_puzzle_results(cli_args)
