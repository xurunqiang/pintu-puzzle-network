"""
模型架构各模块单元测试
"""

import unittest
import torch
from pintu_v2.models.content_encoder import ContentEncoder
from pintu_v2.models.edge_encoder import EdgeEncoder
from pintu_v2.models.retrieval_scorer import RetrievalScorer
from pintu_v2.models.local_fusion import LocalFusion
from pintu_v2.models.global_fusion import GlobalFusion


class TestModelModules(unittest.TestCase):
    def test_content_encoder(self):
        enc = ContentEncoder(content_dim=96)
        x = torch.randn(4, 3, 7, 7)
        feat, pred_pos = enc(x)
        self.assertEqual(feat.shape, (4, 96))
        self.assertEqual(pred_pos.shape, (4, 2))
        self.assertTrue((pred_pos >= 0.0).all() and (pred_pos <= 1.0).all())

        dir_logits = enc.predict_direction(feat, feat)
        self.assertEqual(dir_logits.shape, (4, 5))

    def test_edge_encoder_mask_isolation(self):
        """测试边缘掩码前置隔离与全无效边带检测 (Item 6)"""
        enc = EdgeEncoder(edge_dim=32)
        
        # 1. 正常有效边带
        band = torch.randn(2, 6, 2, 7)
        mask = torch.ones(2, 1, 2, 7)
        feat, val_flag = enc(band, mask)
        self.assertEqual(feat.shape, (2, 32))
        self.assertTrue(val_flag.all())
        norms = torch.norm(feat, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

        # 2. 全无效边带 (mask 全部为 0): 应输出 0 向量且 val_flag 为 False
        zero_mask = torch.zeros(2, 1, 2, 7)
        zero_feat, zero_val = enc(band, zero_mask)
        self.assertFalse(zero_val.any())
        self.assertTrue(torch.allclose(zero_feat, torch.zeros_like(zero_feat)))

        # 3. 屏蔽无效区域: 修改无效区域的像素不应改变最终特征
        part_mask = torch.zeros(1, 1, 2, 7)
        part_mask[:, :, :, :4] = 1.0 # 前 4 列有效，后 3 列无效
        band1 = torch.randn(1, 6, 2, 7)
        band2 = band1.clone()
        band2[:, :, :, 4:] += 100.0 # 剧烈改变无效区域的像素

        feat1, _ = enc(band1, part_mask)
        feat2, _ = enc(band2, part_mask)
        # 前置隔离后，两者的特征应当严格完全一致
        self.assertTrue(torch.allclose(feat1, feat2, atol=1e-5))

    def test_local_and_global_fusion(self):
        local_f = LocalFusion(dim=96)
        global_f = GlobalFusion(dim=96, grid_size=5)

        grid_feat = torch.randn(2, 96, 5, 5)
        grid_mask = torch.ones(2, 1, 5, 5)

        loc_out = local_f(grid_feat, grid_mask, 2, 2)
        self.assertEqual(loc_out.shape, (2, 96))

        glob_out = global_f(grid_feat)
        self.assertEqual(glob_out.shape, (2, 96))


if __name__ == "__main__":
    unittest.main()
