"""
CIFAR-10 数据集加载、划分与两视角数据增强
"""

import os
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T


class CIFAR10PairDataset(Dataset):
    """
    包装 CIFAR-10，返回两组自监督增强视图 (用于全图 VICReg 对比)，
    以及用于切片拼图的原始基准图像 (保持 [0, 1] 像素空间)。
    """
    def __init__(self, root: str, train: bool = True, seed: int = 42, augment: bool = True):
        super().__init__()
        self.augment = augment
        self.raw_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=T.ToTensor()
        )
        
        # 针对 32x32 小图的自监督增强 (视图 1 与 视图 2)
        self.id_offset = 0 if train else 50000
        self.geometry_transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
        ])
        self.photometric_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.1),
        ])

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_img, label = self.raw_dataset[idx] # raw_img: (3, 32, 32) in [0, 1]
        
        # 生成两个用于全图自监督表示学习的增强视图
        # 两个中心视图必须来自同一空间区域，颜色扰动可以不同。
        if self.augment:
            geometry = self.geometry_transform(raw_img)
            view1 = self.photometric_transform(geometry)
            view2 = self.photometric_transform(geometry)
        else:
            view1 = view2 = raw_img

        return {
            "img_id": idx + self.id_offset,
            "raw_img": raw_img,
            "view1": view1,
            "view2": view2,
            "label": label
        }


def get_cifar_dataloaders(
    data_dir: str = "./data/cifar10",
    batch_size: int = 64,
    num_workers: int = 4,
    train_val_split: int = 45000,
    seed: int = 42,
    for_evaluation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    划分 CIFAR-10 为 45k 训练集、5k 验证集和 10k 测试集。
    """
    full_train = CIFAR10PairDataset(root=data_dir, train=True, seed=seed, augment=not for_evaluation)
    test_ds = CIFAR10PairDataset(root=data_dir, train=False, seed=seed, augment=not for_evaluation)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = indices[:train_val_split]
    val_indices = indices[train_val_split:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=not for_evaluation,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=not for_evaluation
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
