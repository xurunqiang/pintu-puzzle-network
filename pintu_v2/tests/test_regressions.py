"""覆盖库存约束、检查点一致性和真实训练/评估路径。"""
import copy
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset, Subset

from pintu_v2.checkpoint import load_model_checkpoint, restore_pool
from pintu_v2.config import ModelConfig
from pintu_v2.data.cifar_dataset import CIFAR10PairDataset
from pintu_v2.data.patch_pool import PatchPool, BatchInventory
from pintu_v2.data.patch_slicer import PatchSlicer
from pintu_v2.evaluate import extract_dataset_features, run_image_retrieval
from pintu_v2.losses.composite_loss import CompositeLoss
from pintu_v2.models.pintu_model import PintuModel
from pintu_v2.models.retrieval_scorer import RetrievalScorer
from pintu_v2.train import evaluate_validation_loss, extract_batch_training_pairs


class TinyImages(Dataset):
    def __init__(self, size=20):
        self.images = torch.rand(size, 3, 32, 32, generator=torch.Generator().manual_seed(901))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return dict(raw_img=image, view1=image, view2=image * 0.9,
                    img_id=index, label=index % 10)


def make_pool(model, size=128):
    pool = PatchPool(size, model.grid_size, seed=11)
    pool.build_pool(TinyImages())
    pool.refresh_features(model.content_encoder, model.edge_encoder)
    return pool


def test_feasible_conflict_assignment_and_masked_gradients():
    scorer = RetrievalScorer()
    scores = torch.tensor([[3., 2., 1.], [3., 7., 10.]], requires_grad=True)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    values = torch.eye(3)
    output, ids, soft = scorer.select_with_straight_through(scores, values, valid)
    assert ids.tolist() == [1, 0]
    assert soft[0, 0] == 0 and soft[1, 1:].sum() == 0
    output[0, 2].backward()
    assert scores.grad[0, 2].abs() > 0
    assert scores.grad[~valid].abs().sum() == 0


def test_exhausted_and_infeasible_masks_fail():
    scorer = RetrievalScorer()
    for valid in (torch.tensor([[False, False], [True, True]]),
                  torch.tensor([[True, False], [True, False]])):
        with pytest.raises(RuntimeError):
            scorer.select_with_straight_through(torch.zeros(2, 2), torch.eye(2), valid)


@pytest.mark.parametrize('grid', [3, 5, 7])
def test_checkpoint_reproduces_pool_features_and_assembly(grid, tmp_path):
    torch.manual_seed(7)
    model = PintuModel(ModelConfig(grid_size=grid)).eval()
    pool = make_pool(model)
    original_features = pool.content_feats.clone()
    center_images = TinyImages(2).images
    ids = torch.tensor([50000, 50001])
    with torch.no_grad():
        before = model.assemble_puzzle(center_images, ids, pool)
    checkpoint = dict(model_state=model.state_dict(), model_config=asdict(model.cfg),
                      pool_state=pool.state_dict(), seed=11,
                      args={'consumed_inventory': True, 'batch_size': 2})
    path = tmp_path / 'model.pth'
    torch.save(checkpoint, path)
    model2, checkpoint2 = load_model_checkpoint(path, torch.device('cpu'))
    pool2 = restore_pool(checkpoint2, model2, TinyImages(), torch.device('cpu'))
    with torch.no_grad():
        after = model2.assemble_puzzle(center_images, ids, pool2)
    torch.testing.assert_close(before['z_puzzle'], after['z_puzzle'], rtol=0, atol=0)
    torch.testing.assert_close(before['puzzle_img'], after['puzzle_img'], rtol=0, atol=0)
    torch.testing.assert_close(original_features, pool.content_feats, rtol=0, atol=0)
    assert after['fusion_count'] == grid * grid
    picked = torch.cat([t['selected_idx'] for t in after['trajectory']])
    assert picked.unique().numel() == 2 * (grid * grid - 1)
    assert after['trajectory'][0]['scores'].shape[-1] == 5


def test_pool_manifest_is_slot_aligned_and_refresh_preserves_mode():
    model = PintuModel(ModelConfig(grid_size=3)).train()
    data = TinyImages()
    subset = Subset(data, [8, 2, 15, 5])
    pool = PatchPool(24, 3, seed=17)
    pool.build_pool(subset)
    for slot, (sub_index, image_id, r, c) in enumerate(pool.sampled_tuples):
        assert image_id == subset.indices[sub_index] == pool.source_img_ids[slot]
        pieces, _ = pool.slicer.slice_image(data.images[image_id:image_id + 1])
        torch.testing.assert_close(pieces[0, r, c], pool.patches[slot])
    pool.refresh_features(model.content_encoder, model.edge_encoder)
    assert model.content_encoder.training and model.edge_encoder.training


def test_optional_mask_and_invalid_gradient_pixels():
    model = PintuModel()
    image = torch.rand(1, 3, 7, 7)
    assert model.encode_patch(image)[2].shape == (1, 4, 32)
    # 外层有效、内层无效；差分不能把无效内层传到有效外层梯度通道。
    mask = torch.ones(1, 1, 7, 7)
    mask[:, :, 1] = 0
    changed = image.clone()
    changed[:, :, 1] = 100
    bands1, masks1 = model.slicer.extract_edge_bands(image, mask)
    bands2, masks2 = model.slicer.extract_edge_bands(changed, mask)
    f1, _ = model.edge_encoder(bands1['top'], masks1['top'])
    f2, _ = model.edge_encoder(bands2['top'], masks2['top'])
    torch.testing.assert_close(f1, f2, rtol=0, atol=0)


def test_masked_seam_normalization_matches_unmasked():
    a, b = torch.rand(2, 6, 2, 7), torch.rand(2, 6, 2, 7)
    mask = torch.ones(2, 1, 2, 7)
    plain = PatchSlicer.compute_raw_seam_difference(a, b)
    masked = PatchSlicer.compute_raw_seam_difference(a, b, mask, mask)
    for x, y in zip(plain, masked):
        torch.testing.assert_close(x, y)


def test_validation_and_evaluation_honor_reuse_mode():
    model = PintuModel(ModelConfig(grid_size=3, include_source_patches=False)).train()
    pool = make_pool(model, size=4)  # 共享消耗无法组装，复用模式应正常通过。
    teacher = copy.deepcopy(model).eval()
    images = TinyImages(2).images
    batch = dict(raw_img=images, view1=images, view2=images * .9,
                 img_id=torch.tensor([50000, 50001]), label=torch.tensor([0, 1]))
    rng_before = torch.random.get_rng_state()
    loss = evaluate_validation_loss(model, teacher, CompositeLoss(), [batch], 'full',
                                    pool, torch.device('cpu'), consumed_inventory=False)
    assert torch.isfinite(torch.tensor(loss))
    assert model.training
    assert torch.equal(rng_before, torch.random.get_rng_state())
    with patch.object(model, 'assemble_puzzle', wraps=model.assemble_puzzle) as assemble:
        features, _ = extract_dataset_features(model, [batch], 'puzzle', pool,
                                               torch.device('cpu'), consumed_inventory=False)
        assert features.shape == (2, 96)
        assert assemble.call_args.kwargs['consumed_inventory'] is False
        assert assemble.call_args.kwargs['record_trajectory'] is False


def test_direction_negatives_are_same_image_and_all_classes_covered():
    model = PintuModel()
    images = torch.arange(10).view(10, 1, 1, 1).expand(10, 3, 32, 32).float() / 10
    seen = []
    handle = model.content_encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach()))
    sample = extract_batch_training_pairs(model, images, torch.device('cpu'))
    handle.remove()
    assert sample['dir_targets'].bincount(minlength=5).tolist() == [2] * 5
    torch.testing.assert_close(seen[1].mean(dim=(1, 2, 3)), seen[2].mean(dim=(1, 2, 3)))


def test_reuse_penalty_has_gradient_with_diffuse_probabilities():
    inventory = BatchInventory(1, 100, torch.arange(100), torch.tensor([1000]), False)
    logits = torch.linspace(-.2, .2, 100, requires_grad=True)
    weights = logits.softmax(0).unsqueeze(0)
    inventory.record_soft_usage(weights)
    inventory.record_soft_usage(weights)
    loss = inventory.get_repetition_loss()
    assert loss > 0
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_test_ids_do_not_collide_and_geometry_is_shared():
    class FakeCIFAR:
        def __init__(self, **kwargs):
            pass
        def __getitem__(self, index):
            return torch.arange(3072).reshape(3, 32, 32).float() / 3072, 0
        def __len__(self):
            return 10
    with patch('torchvision.datasets.CIFAR10', FakeCIFAR):
        train = CIFAR10PairDataset('unused', train=True)
        test = CIFAR10PairDataset('unused', train=False)
    train.geometry_transform = lambda x: x.flip(-1)
    train.photometric_transform = lambda x: x
    sample = train[0]
    torch.testing.assert_close(sample['view1'], sample['view2'])
    assert sample['img_id'] != test[0]['img_id']


def test_small_gallery_never_retrieves_self():
    metrics = run_image_retrieval(torch.eye(2), torch.tensor([0, 1]), max_queries=2)
    assert all(value == 0 for value in metrics.values())


@pytest.mark.parametrize('grid', [3, 5, 7])
@pytest.mark.parametrize('consume', [False, True])
def test_real_backward_reaches_direction_and_local_modules(grid, consume):
    model = PintuModel(ModelConfig(grid_size=grid)).train()
    pool = make_pool(model)
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    output = model.assemble_puzzle(image, torch.tensor([50000, 50001]), pool,
                                  consumed_inventory=consume, record_trajectory=False)
    # LayerNorm 输出的 sum 几乎恒定，不能用于检查非零梯度。
    loss = (output['z_puzzle'] * torch.linspace(-1, 1, 96)).sum() + output['repetition_loss']
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    for module in (model.content_encoder.dir_head, model.local_fusion):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert output['trajectory'] == []
