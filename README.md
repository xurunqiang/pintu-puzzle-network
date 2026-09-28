# Pintu V2 — Patch Puzzle Network

一个用于 CIFAR-10 32×32 小图像的自监督拼图网络实验。模型保留图像中心 Patch，按边缘与语义关系从跨图候选池逐步补全其余位置，再用局部卷积、局部注意力和全局注意力形成图像表示。

支持 `3×3`、`5×5` 和 `7×7` Patch 网格。当前默认网格为 `5×5`。此项目仍在研究阶段；新增评分权重尚未通过完整训练做精度验证。

## 安装

建议使用 Python 3.10 或更新版本，并按本机 CUDA 环境选择合适的 PyTorch 安装包。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

准备 CIFAR-10 二进制数据集，默认放在 `./data/cifar10`，也可以通过 `--data_dir` 指定其他目录。
数据集不包含在本仓库中。

## 训练

```bash
python -m pintu_v2.train --grid_size 5 --device cuda
```

常用参数：

```bash
python -m pintu_v2.train \
  --grid_size 5 \
  --batch_size 64 \
  --pool_size 8192 \
  --inventory_scope image \
  --data_dir ./data/cifar10 \
  --save_dir ./checkpoints/grid5
```

库存选项：

- `--inventory_scope batch`：默认，同一批图片共享并消耗候选库存。
- `--inventory_scope image`：每张图独立消耗，避免同批图片争抢候选块。
- `--no_consumed_inventory`：允许候选复用，并通过使用次数惩罚重复选择。
- `--exclude_self_patches`：排除来自种子图片的候选块。

允许同源候选只表示池中已有的同源 Patch 可以被选择；当前均匀采样的候选池不保证含有种子原图的全部 Patch。本项目实现的是跨图语义补全，并不保证还原原图。

训练默认将检查点写到 `./checkpoints/pintu_v2`。请为不同网格使用不同保存目录。

## 评估与可视化

```bash
python -m pintu_v2.evaluate \
  --checkpoint ./checkpoints/grid5/best_checkpoint.pth \
  --feature_mode all

python -m pintu_v2.visualize \
  --checkpoint ./checkpoints/grid5/best_checkpoint.pth \
  --num_samples 3
```

评估支持 `puzzle`、`full_image`、`center` 和 `all`。可视化会生成中心块对比、组装过程图和接缝热力图。

默认沿用检查点中的库存范围。共享库存的拼图特征受批次大小、分组和顺序影响；用 `--inventory_scope image` 可评估每张图独立检索，但这会改变评估协议。
可视化的 `--num_samples` 只控制展示数量，实际组装批次默认沿用检查点的 batch size；可用 `--batch_size` 显式调整。

## 续训

```bash
python -m pintu_v2.train \
  --grid_size 5 \
  --inventory_scope image \
  --resume ./checkpoints/grid5/latest_checkpoint.pth
```

续训恢复优化模型、动量模型、优化器、学习率调度器、候选池和 PyTorch 随机数状态。训练参数需要与检查点一致；数据目录和输出目录可以不同。仅支持本项目当前版本生成的完整训练检查点。中断 epoch 中尚未保存的批次会重新执行。

## 设计要点

- 中心 Patch 初始化，通过 FIFO 队列逐步填充网格。
- 候选评分结合语义相似度、方向预测、边带特征、真实像素颜色与法向梯度差、边缘有效性和位置先验。
- 局部融合结合掩码卷积和十字窗口注意力；拼图完成后使用全局融合。
- 训练包含全图与拼图双视图自监督目标、动量教师对齐、方向预测和接缝特征学习。
- `ModelConfig` 中的接缝评分参数是待调超参数；正式报告性能前，应在目标硬件上完成训练与消融评估。

## 许可证

本项目使用 MIT License。数据集及第三方组件各自适用其原有许可。
