"""
损失函数单元测试
"""

import unittest
import torch
from pintu_v2.losses.vicreg import VICRegLoss
from pintu_v2.losses.composite_loss import CompositeLoss


class TestLosses(unittest.TestCase):
    def test_vicreg(self):
        v = VICRegLoss()
        z1 = torch.randn(16, 96, requires_grad=True)
        z2 = torch.randn(16, 96, requires_grad=True)
        loss, details = v(z1, z2)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_invar", details)
        self.assertIn("loss_var", details)
        self.assertIn("loss_cov", details)

        loss.backward()
        self.assertIsNotNone(z1.grad)

    def test_composite_loss_warmup(self):
        comp = CompositeLoss()
        B = 16
        z_v1 = torch.randn(B, 96)
        z_v2 = torch.randn(B, 96)
        e_a = torch.randn(B, 32)
        e_b = torch.randn(B, 32)
        dir_logits = torch.randn(B, 5)
        dir_targets = torch.randint(0, 5, (B,))
        pred_pos = torch.rand(B, 2)
        target_pos = torch.rand(B, 2)

        tot_warmup, d_warmup = comp(
            phase="warmup",
            z_v1=z_v1,
            z_v2=z_v2,
            edge_pairs=(e_a, e_b),
            dir_logits=dir_logits,
            dir_targets=dir_targets,
            pred_pos=pred_pos,
            target_pos=target_pos
        )
        self.assertGreater(tot_warmup.item(), 0.0)
        self.assertIn("loss_edge", d_warmup)
        self.assertIn("loss_dir", d_warmup)
        self.assertIn("loss_pos", d_warmup)

    def test_composite_loss_full_dual_puzzle(self):
        """测试解耦的双拼图视图 VICReg 与教师对齐损失"""
        comp = CompositeLoss()
        B = 16
        z_v1 = torch.randn(B, 96)
        z_v2 = torch.randn(B, 96)
        z_puz1 = torch.randn(B, 96, requires_grad=True)
        z_puz2 = torch.randn(B, 96, requires_grad=True)
        z_teacher = torch.randn(B, 96)
        rep_loss = torch.tensor(0.5, requires_grad=True)

        tot_full, d_full = comp(
            phase="full",
            z_v1=z_v1,
            z_v2=z_v2,
            z_puzzle1=z_puz1,
            z_puzzle2=z_puz2,
            z_teacher=z_teacher,
            repetition_loss=rep_loss
        )
        self.assertGreater(tot_full.item(), 0.0)
        self.assertIn("loss_vicreg_puzzle", d_full)
        self.assertIn("loss_align", d_full)
        self.assertIn("loss_rep", d_full)

        tot_full.backward()
        self.assertIsNotNone(z_puz1.grad)
        self.assertIsNotNone(rep_loss.grad)


if __name__ == "__main__":
    unittest.main()
