"""
“拼图”网络第二版：评估程序 (evaluate.py)
支持 puzzle (默认) / full_image / center 三种特征源评估与对照
严格利用验证集特征选择探测器轮次，并在测试集进行标准 mAP 评测
"""

import os
import sys
import argparse
import time
import copy
from typing import Tuple, Dict, List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

from pintu_v2.config import ModelConfig
from pintu_v2.data.cifar_dataset import get_cifar_dataloaders
from pintu_v2.data.patch_pool import PatchPool
from pintu_v2.models.pintu_model import PintuModel
from pintu_v2.checkpoint import (
    load_model_checkpoint, restore_pool, checkpoint_seed,
    checkpoint_inventory_mode, checkpoint_allow_self_patches, checkpoint_inventory_scope
)


def parse_args():
    parser = argparse.ArgumentParser(description="评估拼图网络第二版特征表示质量")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pintu_v2/best_checkpoint.pth", help="模型权重文件路径")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="CIFAR-10 数据目录")
    parser.add_argument("--feature_mode", type=str, default="puzzle", choices=["puzzle", "full_image", "center", "all"], help="特征来源模式")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--inventory_scope", choices=['batch', 'image'], default=None, help="默认沿用检查点；image 消除同批样本竞争")
    parser.add_argument("--probe_epochs", type=int, default=20, help="线性探测器训练轮数")
    parser.add_argument("--probe_lr", type=float, default=1e-2, help="线性探测器学习率")
    parser.add_argument("--allow_self_patches", dest="allow_self_patches", action="store_true", default=None, help="允许挑选自身原图候选块 (拼回原图)")
    parser.add_argument("--exclude_self_patches", dest="allow_self_patches", action="store_false", help="禁止挑选自身原图候选块 (严格同源排除)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--quick_test", action="store_true", help="快速测试模式")
    return parser.parse_args()


@torch.no_grad()
def extract_dataset_features(
    model: PintuModel,
    dataloader: DataLoader,
    feature_mode: str,
    pool: PatchPool,
    device: torch.device,
    max_batches: Optional[int] = None,
    consumed_inventory: bool = True,
    allow_self_patches: bool = True,
    inventory_scope: str = 'batch'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据指定的 feature_mode 提取数据集中图像的 96 维特征向量。
    - puzzle: 经过完整的 24 步队列扩展与候选池检索组装出的 z_puzzle (真实拼图表征)
    - full_image: 完整图像直接切片融合得到的 z_full
    - center: 仅使用中心 Seed Patch 经内容编码器得到的 96 维特征
    """
    model.eval()
    all_feats = []
    all_labels = []

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break
        raw_imgs = batch["raw_img"].to(device)
        img_ids = batch["img_id"].to(device)
        labels = batch["label"]

        if feature_mode == "puzzle":
            res = model.assemble_puzzle(raw_imgs, img_ids, pool, tau=0.2,
                                        consumed_inventory=consumed_inventory,
                                        record_trajectory=False,
                                        allow_self_patches=allow_self_patches,
                                        inventory_scope=inventory_scope)
            feats = res["z_puzzle"]
        elif feature_mode == "full_image":
            feats = model.encode_full_image(raw_imgs)
        elif feature_mode == "center":
            patches, _ = model.slicer.slice_image(raw_imgs)
            cr, cc = model.center_coord
            c_patches = patches[:, cr, cc] # (B, 3, P, P)
            feats, _ = model.content_encoder(c_patches)
        else:
            raise ValueError(f"未知 feature_mode: {feature_mode}")

        all_feats.append(feats.cpu())
        all_labels.append(labels)

    features = torch.cat(all_feats, dim=0) # (N, 96)
    targets = torch.cat(all_labels, dim=0)  # (N,)
    return features, targets


def run_linear_probe_with_val_selection(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu")
) -> float:
    """
    在冻结表示上训练线性分类器。
    严格使用验证集准确率挑选最优 epoch 权重，最后在测试集上评测一次测试准确率 (杜绝测试集泄漏)。
    """
    if epochs < 1:
        raise ValueError("probe_epochs 必须至少为 1")
    print(f"\n--- 线性探测评估 (基于验证集选择最优模型) ---")
    in_dim = train_feats.shape[1]
    num_classes = 10

    norm_train_feats = F.normalize(train_feats, p=2, dim=-1)
    norm_val_feats = F.normalize(val_feats, p=2, dim=-1)
    norm_test_feats = F.normalize(test_feats, p=2, dim=-1)

    train_ds = TensorDataset(norm_train_feats, train_labels)
    val_ds = TensorDataset(norm_val_feats, val_labels)
    test_ds = TensorDataset(norm_test_feats, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classifier = nn.Linear(in_dim, num_classes).to(device)
    optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证集评估
        classifier.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = classifier(x).argmax(dim=-1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        val_acc = v_correct / v_total * 100.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(classifier.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Probe Epoch [{epoch:2d}/{epochs:2d}] -> Val Acc: {val_acc:.2f}% (Best Val: {best_val_acc:.2f}%)")

    # 载入验证集选出的最优权重并在测试集上评价
    classifier.load_state_dict(best_weights)
    classifier.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = classifier(x).argmax(dim=-1)
            t_correct += (preds == y).sum().item()
            t_total += y.size(0)

    test_acc = t_correct / t_total * 100.0
    print(f"★ 测试集最终 Top-1 准确率: {test_acc:.2f}% (由验证集最佳轮次选出)")
    return test_acc


def run_image_retrieval(
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    k_list: Tuple[int, ...] = (1, 5, 10),
    max_queries: int = 1000
) -> Dict[str, float]:
    """
    无监督余弦相似度图像检索评估。
    使用严格标准 mAP 计算: 分母为 min(库中总同类正样本数, K)。
    """
    print(f"\n--- 无监督图像语义检索评估 (Query 数: {min(max_queries, len(test_feats))}) ---")
    norm_feats = F.normalize(test_feats, p=2, dim=-1) # (N, 96)
    labels = test_labels

    num_samples = len(norm_feats)
    if num_samples < 2:
        raise ValueError('图像检索至少需要两个样本')
    num_queries = min(max_queries, num_samples)

    query_feats = norm_feats[:num_queries]
    query_labels = labels[:num_queries]
    gallery_feats = norm_feats
    gallery_labels = labels

    # 统计库中各类别出现频次 (排除 query 自身后的总相关项数)
    label_counts = torch.bincount(gallery_labels)

    # 相似度矩阵: (Q, G)
    sims = torch.matmul(query_feats, gallery_feats.t())
    for q in range(num_queries):
        sims[q, q] = -1e9 # 排除自身

    if any(k < 1 for k in k_list):
        raise ValueError('检索 K 必须为正整数')
    # 所有指标最多使用前 max(50, K) 项，无需完整排序整个图库。
    ranking_size = min(max((50, *k_list)), num_samples - 1)
    sorted_indices = sims.topk(ranking_size, dim=-1, sorted=True).indices

    results = {}
    for k in k_list:
        top_k = sorted_indices[:, :min(k, num_samples - 1)]
        retrieved_labels = gallery_labels[top_k]
        match_mask = (retrieved_labels == query_labels.unsqueeze(1))
        precision_at_k = match_mask.float().mean().item() * 100.0
        results[f"Precision@{k}"] = precision_at_k
        print(f"Image Retrieval Precision@{k}: {precision_at_k:.2f}%")

    # 标准 mAP@50 计算
    k_map = min(50, num_samples - 1)
    top_indices = sorted_indices[:, :k_map] # (Q, k_map)
    top_matches = (gallery_labels[top_indices] == query_labels.unsqueeze(1)).float() # (Q, k_map)
    
    cumulative_correct = torch.cumsum(top_matches, dim=1)
    ranks = torch.arange(1, k_map + 1, dtype=torch.float32, device=test_feats.device).unsqueeze(0)
    precision_at_rank = cumulative_correct / ranks

    # 库中属于该类的总数量 (排除自身)
    total_relevant = (label_counts[query_labels] - 1).float() # (Q,)
    max_possible_recall = torch.clamp(torch.minimum(total_relevant, total_relevant.new_tensor(float(k_map))), min=1.0)

    ap_per_query = (precision_at_rank * top_matches).sum(dim=1) / max_possible_recall
    map_score = ap_per_query.mean().item() * 100.0
    results["mAP@50"] = map_score
    print(f"Image Retrieval mAP@50 (标准分母): {map_score:.2f}%")

    return results


def evaluate_single_mode(mode: str, model: PintuModel, pool: PatchPool, dataloaders, args, device):
    print(f"\n==================================================")
    print(f"  正在评估特征模式: [{mode.upper()}]")
    print(f"==================================================")

    train_loader, val_loader, test_loader = dataloaders
    max_b = 3 if args.quick_test else None

    print(f"正在提取 [{mode}] 训练集特征...")
    train_feats, train_labels = extract_dataset_features(
        model, train_loader, mode, pool, device, max_batches=max_b,
        consumed_inventory=args.consumed_inventory, allow_self_patches=args.allow_self_patches,
        inventory_scope=args.inventory_scope
    )
    print(f"正在提取 [{mode}] 验证集特征...")
    val_feats, val_labels = extract_dataset_features(
        model, val_loader, mode, pool, device, max_batches=max_b,
        consumed_inventory=args.consumed_inventory, allow_self_patches=args.allow_self_patches,
        inventory_scope=args.inventory_scope
    )
    print(f"正在提取 [{mode}] 测试集特征...")
    test_feats, test_labels = extract_dataset_features(
        model, test_loader, mode, pool, device, max_batches=max_b,
        consumed_inventory=args.consumed_inventory, allow_self_patches=args.allow_self_patches,
        inventory_scope=args.inventory_scope
    )

    probe_epochs = 2 if args.quick_test else args.probe_epochs
    # 各特征模式使用相同探测器初始化和训练批次顺序。
    torch.manual_seed(args.evaluation_seed)
    probe_acc = run_linear_probe_with_val_selection(
        train_feats, train_labels,
        val_feats, val_labels,
        test_feats, test_labels,
        epochs=probe_epochs,
        lr=args.probe_lr,
        device=device
    )

    retrieval_res = run_image_retrieval(
        test_feats, test_labels,
        k_list=(1, 5, 10),
        max_queries=100 if args.quick_test else 1000
    )

    return {
        "probe_acc": probe_acc,
        **retrieval_res
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"=== 运行拼图网络第二版特征质量评估 ===")
    print(f"检查点路径: {args.checkpoint}")

    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    seed = checkpoint_seed(checkpoint)
    args.evaluation_seed = seed
    torch.manual_seed(seed)
    args.batch_size = args.batch_size or checkpoint.get('args', {}).get('batch_size', 64)
    if args.quick_test:
        args.batch_size = min(args.batch_size, 4)
    args.consumed_inventory = checkpoint_inventory_mode(checkpoint)
    if args.inventory_scope is None:
        args.inventory_scope = checkpoint_inventory_scope(checkpoint)
    print(f'评估库存范围: {args.inventory_scope}；数据分组种子: {seed}')
    if args.consumed_inventory and args.inventory_scope == 'batch':
        print('批次共享评估：结果依赖 batch_size、分组及顺序；可视化复现需要相同批次。')
    if args.allow_self_patches is None:
        args.allow_self_patches = checkpoint_allow_self_patches(checkpoint)
    print(f"网格: {model.grid_size}×{model.grid_size}, 批次: {args.batch_size}, 消耗库存: {args.consumed_inventory}, 允许拼回原图: {args.allow_self_patches}")
    train_loader, val_loader, test_loader = get_cifar_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=2,
        seed=seed, for_evaluation=True,
    )
    pool = None
    if args.feature_mode in ('puzzle', 'all'):
        pool = restore_pool(checkpoint, model, train_loader.dataset, device)
        print(f"候选池已恢复，共 {pool.pool_size} 个候选。")

    modes_to_run = ["puzzle", "full_image", "center"] if args.feature_mode == "all" else [args.feature_mode]
    all_results = {}

    for mode in modes_to_run:
        res = evaluate_single_mode(mode, model, pool, (train_loader, val_loader, test_loader), args, device)
        all_results[mode] = res

    print(f"\n==================== 评估总汇对照表 ====================")
    print(f"{'特征模式':<12} | {'Top-1 Acc':<10} | {'P@1':<8} | {'P@5':<8} | {'mAP@50':<8}")
    print("-" * 55)
    for m, r in all_results.items():
        print(f"{m:<12} | {r['probe_acc']:>8.2f}% | {r['Precision@1']:>6.2f}% | {r['Precision@5']:>6.2f}% | {r['mAP@50']:>6.2f}%")
    print(f"========================================================")


if __name__ == "__main__":
    main()
