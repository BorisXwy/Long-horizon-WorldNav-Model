# GigaNav Ablation：Wan2.1-1.3B 导航模型

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-MDL-002` |
| 类型 | Ablation 模型与训练接口 |
| 状态 | Active / Full-chain implemented |
| 更新时间 | 2026-09-03 |
| 职责 | 记录 GigaWorld-Policy 风格导航 ablation 的结构、输入输出、权重和运行入口 |

## 当前结论与入口

`GigaNavModel` 已在 `src/giga_nav/` 中实现：使用服务器已有的官方
`Wan2.1-T2V-1.3B` 权重作为唯一视频 DiT 初始化，采用 GigaWorld-Policy 的
state/reference/action/future token 排列和 action-only 读出，导航输出改为 R2R
离散四类动作的交叉熵（Cross-Entropy, CE）。完整真实链路的一步 forward、backward
和 optimizer update 已通过：

```text
log/giga_nav_smoke_20260902_retry2/
  config_and_structure.json
  train.jsonl
  step_000001.pt
```

运行入口：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
bash scripts/smoke_giga_nav.sh                         # 完整一批真实 R2R 数据
python scripts/train_giga_nav.py --device cuda:1      # 正式训练入口
python scripts/eval_giga_nav.py --checkpoint <ckpt> \
  --device cuda:1 --batches 20 --output result/giga_nav/<name>.json
```

## 设计边界：保留项与唯一必要改动

GigaWorld-Policy 的公开设计使用 Wan2.2-TI2V-5B 作为共享视频/动作骨干，并将
state、reference visual、action 和 noisy future visual 放进同一个 token stream；
视频/动作训练采用 flow matching，policy-only 推理时只读 action token 的输出。
本 ablation 保留上述 token 顺序、共享 Wan block、文本 cross-attention、clean
prefix/noisy future 因果分段和 action-only readout，只做以下与 R2R 数据及本地
权重必须一致的改动：

1. **Backbone**：Wan2.2-TI2V-5B（约 5B）替换为本地 Wan2.1-T2V-1.3B；因此
   latent channel 从 48 变为 Wan2.1 原生的 16，hidden dim 为 1536、FFN dim
   为 8960、30 层、12 heads。
2. **Action target**：Giga 原生连续 14-D action flow 改为项目统一的 R2R 四类
   离散动作：`STOP`、`MOVE_FORWARD`、`TURN_LEFT`、`TURN_RIGHT`，只改变输出头和
   loss，不另建视觉 backbone。
3. **State**：R2R 没有与 Giga 机器人相同的 proprioceptive state，因此保留
   一个 state token 的位置，输入显式零向量 `[B,1,14]`，避免把缺失信息伪装成
   观测。

这不是 V1 正式 Register/3D/视频生成主线；它是单独的 Giga-style navigation
ablation，用来回答“共享 Wan backbone 直接改造成 policy 是否能学会导航”。

## Backbone、参数和权重

### 参数统计（代码实测）

| 部分 | 参数量 |
| --- | ---: |
| Wan2.1-1.3B shared backbone（含其原生 patch/time/text/head） | 1,421,370,432 |
| 新增 state/action projector + 4 类 policy head | 1,016,324 |
| GigaNavModel 总计 | **1,422,386,756** |

统计脚本写入 `log/giga_nav_smoke_20260902_retry2/config_and_structure.json`，口径为
`sum(parameter.numel())`，未把 optimizer states 计入。

### 初始化策略

```text
/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
```

官方 safetensors 的 825 个 shape-compatible 张量被加载到 837 个 Wan 参数键；
新增 state projector、action projector、token role/position 参数和 navigation
classifier 随机初始化。不会加载 GigaWorld-Policy-0/0.5 的 Wan2.2 checkpoint，
因为其 latent channel、hidden width 和 block parameterization 不兼容。

同时使用同一 Wan2.1 体系的 `UMT5` embedding 输入；本 ablation 不在 forward 内
重复加载 VAE，而是读取已经落盘的 T4 latent。

## 观测、token 和张量接口

### 训练样本中的观测

R2R 渲染和 latent 规则沿用项目 canonical loader：

```text
obs_latent       [B, 16, 4, 56, 112]   # 一个 T_latent=4 的当前观测 chunk
text_embedding   [B, 1, 512, 4096]    # UMT5；指令缺失时使用 empty embedding
text_mask        [B, 512]
state            [B, 1, 14]           # R2R 无 proprioception，显式全 0
action_noise     [B, 8, 14]           # 当前导航设置；Giga 原生 p=48
action_target    [B, 8]               # 四类 R2R action id
action_loss_mask [B, 8]
```

`obs_latent` 的第一个 temporal latent 是 clean reference visual，后三个 temporal
latent 是 future visual slots；policy-only ablation 默认将后三个 slots 置零，不让
缺乏真实未来标签的 R2R 样本产生伪视频监督。完整数据来自：

```text
NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/
/sharedata/NAV/derived/v1/vln/rendered_obs/
  stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/
```

### Giga-style 主序列

Wan block 接收的顺序为：

```text
[ state(1) | reference visual patches | action slots(8) | noisy future visual patches ]
```

其中：

- reference visual 是 `obs_latent[:, :, :1]` 经 Wan 原生 `Conv3d(16 -> 1536,
  kernel=stride=(1,2,2))` 后的 spatial tokens；
- noisy future visual 是其余 3 个 latent temporal planes 的零占位（可由接口传入
 真实 latent 做后续视频辅助 ablation）；
- state/action 通过 Giga 风格 MLP projector 投影到 1536 维后插入同一序列；
- UMT5 text 经过 Wan text cross-attention 投影到 1536 维；
- action hidden 只取 action slots 对应的最终 shared Wan hidden，不经过独立视觉
  encoder。

Wan 的 clean-prefix/noisy-future 分段使 action tokens 能读 state 和 reference，
而不能读 future visual；future visual 可以读取前面的条件和 action slots。这是
Giga action-only 推理所需的因果方向。

### 输出与监督

```text
shared_action_hidden [B,8,1536]
  -> LayerNorm + Linear(1536,4)
  -> action_logits [B,8,4]
  -> masked CE(action_logits, action_target)
```

训练时反向梯度会经过 policy head、action/state projector、Wan action token、共享
Wan blocks、patch/text/time 模块和 visual patch stem；因此不是“冻结 backbone +
旁路小 head”的结果。推理时只执行一次 shared Wan forward 并直接读 8 个 action
slot，跳过 video unpatchify/head。

## 训练设定

默认训练设置保留 GigaWorld-Policy-0 的优化量纲（`lr=6e-5`、`weight_decay=1e-2`），
导航 action horizon 固定为 `8`；Giga 原生 `p=48` 作为参考配置保留在接口中。
当前正式训练使用物理 `batch_size=1`、`gradient_accumulation_steps=32`，因此有效
batch size（EBS）为 **32**；模型结构和每个 micro-batch 的计算不因梯度累积改变。
`weight_decay=1e-2`、物理 `batch_size` 和累积步数均由命令行指定。由于当前环境不保证
`CAME8Bit`，本地入口使用等价可复现的 `AdamW`；这属于 optimizer 实现替换，
不改变 token 或 loss 结构。训练样本由 `GigaNavR2RBatchBuilder` 从 canonical
R2R loader 逐批抽取，并保持 history/window 的真实采样，不复制 latent 文件。

正式训练通过 `scripts/run_giga_nav_train.sh` 启动。该入口固定激活
`virtual_env/.venv_infinite_world`，优先使用 `flash_attn`；若运行环境没有可用的
FlashAttention，Wan DiT 自动退回 PyTorch SDPA（`sdpa_fallback`），不会改变模型结构。

正式长训示例：

```bash
bash scripts/run_giga_nav_train.sh \
  --device cuda:0 --batch-size 1 --action-horizon 8 \
  --grad-accumulation-steps 32 --steps 6000 --save-interval 1000 \
  --output log/giga_nav_wan21_h8_ebs32
```

其中 `CUDA_VISIBLE_DEVICES=1` 时进程内的 GPU1 映射为 `cuda:0`；不要同时把物理
GPU0 的既有任务迁移到这个训练进程。

## 验证状态与限制

已验证：

- 真实 Wan2.1-1.3B 权重加载（825 keys）；
- 真实 R2R latent、instruction embedding 和 action window 读取；
- 完整 Wan forward、action hidden readout、masked CE、反向传播和 optimizer update；
- 代码实测模型参数量及接口几何记录。
- checkpoint 恢复后 action-only inference 和 masked accuracy 入口通过一批真实窗口
  （`result/giga_nav/smoke_eval_step1.json`）；该随机/一步 checkpoint 的 66.67%
  只证明推理链路，不代表训练后导航指标。

尚未验证：

- 长训练后的 R2R closed-loop success rate；
- Giga 原生连续 action flow 与离散 R2R head 的公平数值对比；
- 将真实 future visual latent 接回并进行 video/action co-training。

## 外部实现对应关系

- [GigaWorld-Policy 论文](https://arxiv.org/abs/2603.17240)：Giga 的共享 token
  排列、p=48 action horizon、视频/动作 flow matching 和 action-only 推理依据。
- [GigaWorld-Policy-0.5 model card](https://huggingface.co/open-gigaai/Giga-World-Policy-0.5)：
  Giga 0.5 的公开参数规模与 Wan2.2 5B/MoT 配置依据。
- [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1)：本地官方 Wan2.1
  DiT/VAE/UMT5 初始化来源。
