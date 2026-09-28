"""
数据切片、候选池及批次库存单元测试
"""

import unittest
import torch
from pintu_v2.data.patch_slicer import PatchSlicer
from pintu_v2.data.patch_pool import PatchPool, BatchInventory
from pintu_v2.data.cifar_dataset import get_cifar_dataloaders


class TestDataComponents(unittest.TestCase):
    def test_patch_slicer_3x3(self):
        slicer = PatchSlicer(grid_size=3)
        img = torch.randn(2, 3, 32, 32)
        patches, masks = slicer.slice_image(img)
        self.assertEqual(patches.shape, (2, 3, 3, 3, 11, 11))
        self.assertEqual(masks.shape, (2, 3, 3, 1, 11, 11))

    def test_patch_slicer_5x5(self):
        slicer = PatchSlicer(grid_size=5)
        img = torch.randn(2, 3, 32, 32)
        patches, masks = slicer.slice_image(img)
        self.assertEqual(patches.shape, (2, 5, 5, 3, 7, 7))
        self.assertEqual(masks.shape, (2, 5, 5, 1, 7, 7))
        
        # 验证有效掩码: 原图中心区域应全为 1，外部反射边缘部分为 0
        self.assertEqual(masks[0, 2, 2, 0].mean().item(), 1.0)

        # 验证边带提取
        p = patches[:, 2, 2] # (2, 3, 7, 7)
        m = masks[:, 2, 2]
        bands, b_masks = slicer.extract_edge_bands(p, m)
        for side in ['top', 'right', 'bottom', 'left']:
            self.assertEqual(bands[side].shape, (2, 6, 2, 7))
            self.assertEqual(b_masks[side].shape, (2, 1, 2, 7))

    def test_patch_pool_subset_isolation(self):
        """测试候选池严格从 train_subset 构建，且与验证集完全无交集 (Item 1)"""
        tl, vl, _ = get_cifar_dataloaders(batch_size=8, seed=42)
        train_subset = tl.dataset
        val_subset = vl.dataset

        pool = PatchPool(pool_size=64, grid_size=5, seed=42)
        pool.build_pool(train_subset)

        val_real_ids = {val_subset[i]["img_id"] for i in range(len(val_subset))}
        pool_source_ids = set(pool.source_img_ids.tolist())
        self.assertTrue(pool_source_ids.isdisjoint(val_real_ids))

    def test_batch_shared_inventory(self):
        """测试整批共享库存消耗与耗尽报错机制 (Item 3)"""
        B = 4
        pool_size = 16
        source_ids = torch.tensor([10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 11, 22, 33, 44, 55, 66])
        seed_ids = torch.tensor([10, 20, 33, 99])

        # 默认模式: allow_self_patches=True (允许拼回原图)
        inv_allow = BatchInventory(
            batch_size=B,
            pool_size=pool_size,
            source_img_ids=source_ids,
            seed_img_ids=seed_ids,
            consumed_inventory=True,
            allow_self_patches=True
        )
        valid_mask_allow = inv_allow.get_valid_mask()
        # 同源块不被屏蔽
        self.assertTrue(valid_mask_allow[0, 0].item())
        self.assertTrue(valid_mask_allow[0, 5].item())

        # 消融对照模式: allow_self_patches=False (严格同源排除)
        inv = BatchInventory(
            batch_size=B,
            pool_size=pool_size,
            source_img_ids=source_ids,
            seed_img_ids=seed_ids,
            consumed_inventory=True,
            allow_self_patches=False
        )

        valid_mask = inv.get_valid_mask() # (4, 16)
        # sample 0 seed_id is 10: pool indices 0 and 5 should be False (excluded)
        self.assertFalse(valid_mask[0, 0].item())
        self.assertFalse(valid_mask[0, 5].item())
        self.assertTrue(valid_mask[0, 1].item())

        # 批次共享消耗测试: sample 0 消耗 patch 1
        sel_idx = torch.tensor([1, 2, 3, 4])
        inv.consume(sel_idx)
        valid_mask_after = inv.get_valid_mask()
        
        # 整批共同消耗: 对所有样本，patch 1 均变为不可用
        for b in range(B):
            self.assertFalse(valid_mask_after[b, 1].item())

        # 耗尽测试
        inv.consumed_mask[:] = True
        with self.assertRaises(RuntimeError):
            inv.check_availability()


if __name__ == "__main__":
    unittest.main()
