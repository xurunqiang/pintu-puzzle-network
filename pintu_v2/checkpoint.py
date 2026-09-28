"""训练、评估和可视化共享的模型/候选池恢复协议。"""

import copy
import warnings
from dataclasses import fields

import torch

from pintu_v2.config import ModelConfig
from pintu_v2.data.patch_pool import PatchPool
from pintu_v2.models.pintu_model import PintuModel


def load_model_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('implementation_version') != 3:
        warnings.warn('正在加载旧版权重：本副本改进了检索和接缝评分，输出不保证复现旧版；正式比较建议重新训练。')
    state = checkpoint.get('model_state', checkpoint)
    config = dict(checkpoint.get('model_config', {}))
    config.setdefault('grid_size', checkpoint.get('grid_size',
                      checkpoint.get('args', {}).get('grid_size', state['global_fusion.pos_embed'].shape[-1])))
    config.setdefault('content_dim', state['content_encoder.norm.weight'].numel())
    config.setdefault('edge_dim', state['edge_encoder.proj.weight'].shape[0])
    allowed = {field.name for field in fields(ModelConfig)}
    model = PintuModel(ModelConfig(**{k: v for k, v in config.items() if k in allowed})).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def restore_pool(checkpoint, model, train_dataset, device):
    if 'pool_state' in checkpoint:
        pool = PatchPool.from_state_dict(checkpoint['pool_state'], device=device)
        if pool.grid_size != model.grid_size:
            raise ValueError('候选池与模型网格大小不一致')
    else:
        warnings.warn('旧检查点未保存候选缓存，将用保存的种子和编码器重建；无法保证复现旧缓存。')
        args = checkpoint.get('args', {})
        pool = PatchPool(checkpoint.get('pool_size', args.get('pool_size', 8192)),
                         model.grid_size, device=device,
                         seed=checkpoint.get('seed', args.get('seed', 42)))
        pool.build_pool(train_dataset)
        encoder_model = model
        if 'momentum_model_state' in checkpoint:
            encoder_model = copy.deepcopy(model)
            encoder_model.load_state_dict(checkpoint['momentum_model_state'])
        pool.refresh_features(encoder_model.content_encoder, encoder_model.edge_encoder)
    # 同一训练划分的全局 ID；不读取增强视图，也不消耗随机数。
    if hasattr(train_dataset, 'indices'):
        allowed_ids = set(int(i) for i in train_dataset.indices)
        if not set(pool.source_img_ids.cpu().tolist()).issubset(allowed_ids):
            raise ValueError('检查点候选池来源不属于当前训练划分，请使用检查点中的 seed 和数据集')
    return pool


def checkpoint_inventory_mode(checkpoint):
    return checkpoint.get('args', {}).get('consumed_inventory', True)


def checkpoint_inventory_scope(checkpoint):
    return checkpoint.get('args', {}).get('inventory_scope', 'batch')


def checkpoint_allow_self_patches(checkpoint):
    return checkpoint.get('args', {}).get('allow_self_patches', True)


def checkpoint_seed(checkpoint):
    return checkpoint.get('seed', checkpoint.get('args', {}).get('seed', 42))
