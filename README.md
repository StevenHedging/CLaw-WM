# Continual Multi-Head World Model

这是一个面向研究原型的 PyTorch code base，用于探索 **Continual Multi-Head World Model with Frame-wise Dynamic Head Selection**：世界模型维护一个可增长的动力学头库，并在每一帧或每隔 `m` 帧动态选择动力学头，以适应环境动力学的突然变化。

本项目优先保证 pipeline、模块边界、head selection、head spawning、replay/distillation 接口全部跑通；当前实现不是为了追求 SOTA。

## 目标

给定上下文窗口：

```math
C_t=(I_{t-K+1},\dots,I_t)
```

表示映射器估计当前 latent state：

```math
\hat z_t=E_\theta(C_t)
```

维护动力学头库：

```math
\mathcal{F}_n=\{F_\phi^1,\dots,F_\phi^{M_n}\}
```

每一帧或每隔 `m` 帧选择一次动力学头：

```math
i_t=\pi(C_t,\mathcal{F}_n)
```

预测下一状态与下一帧：

```math
\hat z_{t+1}=F_\phi^{i_t}(\hat z_t), \qquad
\hat I_{t+1}=R_\psi(\hat z_{t+1})
```

整体计算图在 `src/models/world_model.py` 中显式实现为：

```math
\hat I_{t+1}
=
R_\psi
\left(
F_\phi^{\pi(C_t,\mathcal{F}_n)}
\left(
E_\theta(C_t)
\right)
\right)
```

注意：这里的 head selection 是 **frame-wise or every-m-frames**，由 `model.select_every_m` 控制；它不是“每个 task 只选择一次 head”。

## 目录结构

```text
configs/                 # OmegaConf/Hydra-style YAML config
src/
  data/                  # toy physics dataset and dataloader factory
  models/                # encoder, dynamics heads, selector, renderer, world model
  continual/             # head manager, replay buffer, CL losses and regularizers
  training/              # train/eval loops and checkpoint helpers
  utils/                 # seed, logging, visualization
scripts/
  train_toy.py
  eval_toy.py
  visualize_rollout.py
tests/                   # pytest coverage for core modules
```

## 安装

建议使用 Python 3.10+：

```bash
pip install -r requirements.txt
```

`wandb` 是可选依赖，默认 `use_wandb: false`。

## 训练 toy experiment

```bash
python scripts/train_toy.py --config configs/default.yaml
```

训练流程会：

- 初始化 encoder、renderer 和至少一个 dynamics head；
- 读取带突变动力学的 streaming toy physics samples；
- 对每个 batch 评估已有 heads；
- 若最佳误差低于阈值 `continual.reuse_threshold`，复用并更新旧 head；
- 若误差超过阈值且未达到 `continual.max_heads`，扩展新 head；
- 使用 replay loss、distillation loss 和 L2 regularization placeholder 巩固旧知识；
- 保存 checkpoint 到 `outputs/checkpoints/last.pt`。

## 评估

```bash
python scripts/eval_toy.py --config configs/default.yaml
```

若 checkpoint 存在，脚本会自动加载；否则会评估 fresh model。

## 可视化 rollout

```bash
python scripts/visualize_rollout.py --config configs/default.yaml
```

默认输出：

```text
outputs/rollout_comparison.png
```

## 当前实现

- `RepresentationMapper`：共享 CNN + temporal mean pooling。
- `ResidualMLPDynamicsHead`：`F_i(z)=z+dt*f_i(z)`。
- `DynamicsHeadLibrary`：支持 `add_head(init_from=...)`。
- `ErrorBasedHeadSelector`：基于 one-step prediction error 的 hard selection。
- `Renderer`：小型 MLP decoder。
- `HeadManager`：根据误差阈值复用或扩展 head。
- `ReplayBuffer`：保存旧样本及对应 head id。
- 持续学习损失：new prediction loss + replay loss + distillation loss + L2 placeholder。

## 限制

- 当前 head selection 以 batch 平均 one-step error 选一个 head；更细粒度的 per-sample selector 可作为下一步扩展。
- renderer 是 MLP decoder，适合小图 toy setting，不适合真实视频。
- distillation 使用训练过程中的 teacher snapshot，接口已打通但未实现复杂 teacher scheduling。
- OWL-style update 目前以 `init_from` warm start 和 fine-tuning hook 形式预留。
- toy physics 只有单球 2D dynamics，主要用于验证 continual pipeline。

## TODO

- 增加 learned/gated head selector 和 uncertainty-aware selection。
- 支持 per-sample head selection 与多 head soft mixture。
- 引入 object-centric encoder 或 slot encoder。
- 增加更丰富的动力学突变：碰撞、摩擦、外力场、多物体交互。
- 实现更严格的 replay sampling、EWC/SI/MAS regularization 和 teacher checkpoint scheduling。
- 加入 rollout loss 训练项与更完整的评估指标。
