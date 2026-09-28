"""
端到端拼图模型前向、完整局部融合覆盖与反向梯度测试
"""

import unittest
import torch
from pintu_v2.models.pintu_model import PintuModel
from pintu_v2.data.patch_pool import PatchPool
from pintu_v2.config import ModelConfig


class TestPintuModelPipeline(unittest.TestCase):
    def test_full_pipeline_5x5(self):
        B = 2
        grid_size = 5
        pool_size = 64

        model = PintuModel(ModelConfig(grid_size=grid_size))
        pool = PatchPool(pool_size=pool_size, grid_size=grid_size)

        # 构造微型候选池
        P = pool.patch_size
        pool.patches = torch.randn(pool_size, 3, P, P)
        pool.patch_masks = torch.ones(pool_size, 1, P, P)
        pool.edge_bands = torch.randn(pool_size, 4, 6, 2, P)
        pool.edge_masks = torch.ones(pool_size, 4, 1, 2, P)
        pool.edge_valid_flags = torch.ones(pool_size, 4, dtype=torch.bool)
        pool.source_img_ids = torch.arange(100, 100 + pool_size)
        pool.source_coords = torch.rand(pool_size, 2)
        pool.refresh_features(model.content_encoder, model.edge_encoder)

        seed_imgs = torch.randn(B, 3, 32, 32, requires_grad=True)
        seed_ids = torch.tensor([1, 2])

        result = model.assemble_puzzle(seed_imgs, seed_ids, pool, tau=1.0)
        
        # 验证输出维度与完整覆盖
        self.assertEqual(result["z_puzzle"].shape, (B, 96))
        self.assertEqual(result["puzzle_img"].shape, (B, 3, 35, 35))
        self.assertEqual(len(result["trajectory"]), 24) # 5x5 - 1 = 24 步
        self.assertEqual(result["fusion_count"], 25)    # 25 个位置全部作为中心局部融合 (Item 4)

        # 验证梯度反向传播
        loss = result["z_puzzle"].sum()
        loss.backward()
        self.assertIsNotNone(seed_imgs.grad)
        self.assertFalse(torch.isnan(seed_imgs.grad).any())

    def test_full_pipeline_3x3(self):
        """测试 3x3 网格下的兼容性 (Item 7)"""
        B = 2
        grid_size = 3
        pool_size = 16

        model = PintuModel(ModelConfig(grid_size=grid_size))
        pool = PatchPool(pool_size=pool_size, grid_size=grid_size)

        P = pool.patch_size
        pool.patches = torch.randn(pool_size, 3, P, P)
        pool.patch_masks = torch.ones(pool_size, 1, P, P)
        pool.edge_bands = torch.randn(pool_size, 4, 6, 2, P)
        pool.edge_masks = torch.ones(pool_size, 4, 1, 2, P)
        pool.edge_valid_flags = torch.ones(pool_size, 4, dtype=torch.bool)
        pool.source_img_ids = torch.arange(100, 100 + pool_size)
        pool.source_coords = torch.rand(pool_size, 2)
        pool.refresh_features(model.content_encoder, model.edge_encoder)

        seed_imgs = torch.randn(B, 3, 32, 32)
        seed_ids = torch.tensor([1, 2])

        result = model.assemble_puzzle(seed_imgs, seed_ids, pool, tau=1.0)
        self.assertEqual(result["puzzle_img"].shape, (B, 3, 33, 33))
        self.assertEqual(len(result["trajectory"]), 8) # 3x3 - 1 = 8 步
        self.assertEqual(result["fusion_count"], 9)    # 9 个位置全部局部融合


if __name__ == "__main__":
    unittest.main()
