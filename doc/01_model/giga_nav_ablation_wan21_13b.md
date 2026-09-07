# GigaNav Ablation：Wan2.1-1.3B 导航模型

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-MDL-002` |
| 类型 | Ablation 模型与训练接口 |
| 状态 | Active / policy-video 三模式 cotrain 已实现并进入完整训练验证 |
| 更新时间 | 2026-09-07 |
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
# Habitat 与 GigaNav 分属两个虚拟环境，由该入口自动启动完整推理服务
cd ../StreamVLN
CUDA_VISIBLE_DEVICES=0 ../virtual_env/.venv_streamvln/bin/python \
  ../NAV/scripts/eval_giga_nav_closed_loop.py \
  --checkpoint ../NAV/log/<run>/step_005000.pt \
  --split train --max-episodes 20 \
  --output ../NAV/result/giga_nav/<name>
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
| 新增 state/action projector + 4 类 policy head（H=8） | 954,884 |
| GigaNavModel 总计 | **1,422,325,316** |

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

### 当前实际速度

在同一张 RTX 6000 Ada、同一真实 R2R batch、H=8 的完整 Wan forward/backward
路径上测得：虚拟环境 `torch 2.6.0+cu124 + flash_attn 2.7.4.post1` 的单
micro-step 为 1.508s（首次）以及 1.280s、1.208s（warm，平均约 1.24s）；系统
环境的 `sdpa_fallback` 为 2.126s（首次）以及 1.242s、1.239s（warm，平均约
1.24s）。因此 FlashAttention 已正确启用，但在当前 token 几何下端到端总时延主要
由 Wan 其他模块、数据和反向计算决定，不能把 attention kernel 的局部加速直接等同
为同等比例的训练加速。

正式 EBS32（32 个 micro-step）实测 optimizer step 为 38.67s、40.85s；峰值显存
约 48.3 GiB。启动器设置 `expandable_segments:True` 以避免第二个 optimizer step
的临时 RoPE 张量因显存碎片触发 OOM。

## 已有训练对应的闭环时间语义

已有 `step_005000.pt` 不应改标签重训。它的 canonical dataloader 使用：

```text
T4 window frames = [12m, ..., 12m+12]
policy-visible reference = first causal latent plane = frame 12m
action target = [A(12m+12), ..., A(12m+19)]
```

因此它是一个 **12-step look-ahead / delayed-observation policy**。此前闭环脚本错误地
把当前 RGB 直接编码为 reference 并立即执行首动作，相当于把训练时的 `+12` 延迟静默
删掉。现在保持 checkpoint 与训练代码不变，闭环改为维护 13 帧滚动 observation
window：环境时刻 `t` 输入 `[O(t-12), ..., O(t)]`，Wan VAE 编为 T4；模型可见的首
plane 是 `O(t-12)`，首个输出正好作为 `A(t)` 执行。由于现有
`forward_policy()` 会丢弃后三个 latent plane 并改成全零，server 实际只编码滚动
窗口最旧的 `O(t-12)`，再补三个零 plane；这与完整编码 13 帧后进入当前 policy 的
有效张量严格相同，但避免每个环境 step 重算无效的 12 帧 VAE。

| 项目 | 已有训练 | 修正后的闭环推理 |
| --- | --- | --- |
| RGB 时间窗口 | 13 帧、stride 12 | 过去至当前的滚动 13 帧 |
| policy-visible plane | 窗口第一帧的 causal latent | `O(t-12)` 的 causal latent |
| 动作起点 | 窗口起点 `+12` | 当前时刻 `t` |
| 执行方式 | H=8 CE supervision | 执行首动作并逐环境 step 重规划 |

前 12 个环境 step 尚无完整过去窗口，脚本以 episode 首帧左填充作为 causal bootstrap；
这 12 步会单独标记，且不纳入 early/middle/late 的对齐统计。该 bootstrap 是已有
look-ahead checkpoint 在 episode 起点不可避免的边界限制，不伪装成完全对齐样本。

Wan causal VAE 数值审计表明：相同首帧时，单帧 encode 与带任意后续 12 帧的 13 帧
encode 首 plane 完全相同（max/mean absolute difference 均为 0）。这也确认已有模型
实际上只利用滚动窗口最旧的 reference frame；后续三 plane 在 policy forward 中仍
显式置零。【已验证→`result/giga_nav/reference_frame_alignment_v2_20260905/vae_causal_equivalence.json`】

误开的 `giga_nav_wan21_h8_ebs32_alignment_v2_20260905` 已在 step13 后停止，没有产生
正式 checkpoint，也不再作为后续方案；训练与 dataloader 默认值均恢复为已有
`step_005000.pt` 的原始口径。

## 旧版“当前单帧立即执行”闭环诊断

2026-09-05 使用旧版 `step_005000.pt` 在 R2R `train` split 的前 20 条 episode 上完成
Habitat 在线闭环评测。每个环境 step 都从当前 RGB 经 Wan causal VAE 得到一个
clean reference latent，补三个全零 future latent plane 以保持 `[B,16,4,56,112]`
接口；模型预测 H=8，但只执行第一个动作，随后读取新 observation 重新规划。

单帧 encode 与训练缓存的第一 latent plane 语义一致：Wan VAE 的 causal encode
明确按 `[1,4,4,...]` 帧分组，第一 latent 只由第一帧产生；同时当前
`GigaNavModel.forward_policy()` 只读取 `obs_latent[:,:,:1]`，后三个 temporal plane
被显式置零。因此该评测没有省略 policy 实际可见的视觉信息，也暴露出此 ablation
实际上没有使用 13 帧 observation chunk 或历史 Register。

| 指标（20-episode 子集） | 结果 |
| --- | ---: |
| Success Rate (SR) | **0.0%** |
| SPL | **0.0%** |
| Oracle Success | **0.0%** |
| Navigation Error (NE) | **10.519 m** |
| 平均环境步数 | **477.5** |
| 总动作数 | 9,550 |
| STOP / FORWARD / LEFT / RIGHT | 1 / 8,025 / 752 / 772 |

动作比例为 STOP 0.01%、MOVE_FORWARD 84.03%、TURN_LEFT 7.87%、TURN_RIGHT 8.08%。
只有 1 条 episode 主动 STOP（第 50 步、NE=4.49 m），其余大多跑满 500 步。
主要失败模式是持续前进撞墙，或 TURN_LEFT/TURN_RIGHT 周期；20 条轨迹均未进入过
成功半径，说明失败不只是 STOP calibration，而是缺少能打破 observation-action
循环的历史状态与闭环恢复能力。

推理使用相同 RGB 的精确哈希缓存：这只复用同一 episode 内完全相同 observation
对应的确定性完整模型输出，不改变动作。9,550 步中 9,034 次为缓存命中，516 次
执行新前向。对 516 次新前向按 step 加权统计：预处理约 3.05 ms、Wan VAE 约
62.21 ms、GigaNav policy forward 约 476.68 ms。结果文件：
`result/giga_nav/step5000_r2r_train_closed_loop20_20260905/{summary.json,episodes.jsonl}`。

该结果与同一 checkpoint 的训练窗口开环 Accuracy 91.50%、Macro-F1 88.86% 形成
直接反差。现在确认原因之一是评测删掉了训练定义的 +12-step 时间关系，因此这组
SR=0 结果只保留为错误推理协议诊断，不能作为已有 checkpoint 的最终闭环结果。
【已验证→`result/giga_nav/step5000_r2r_train_closed_loop20_20260905/summary.json`】

## 2026-09-07：GigaPolicy-0.5 风格 video cotrain 扩展

### 三种可分离训练模式

同一个完整 `GigaNavModel` 现在支持通过 YAML 与统一脚本选择三种目标，不复制
backbone，也不建立旁路小模型：

| 模式 | Action slots 输入 | Video head | Policy head | Loss |
| --- | --- | --- | --- | --- |
| `video_only` / AC-WM | GT 离散动作 one-hot 写入 14-D action input | 开启 | 计算但不监督 | `L_video` |
| `policy_only` | 全零 action query，保持既有 H=8 policy 定义 | 关闭 | 开启 | `L_CE` |
| `cotrain` / mixed AC-WM+WAM | AC-WM batch 用 GT；WAM batch 用全零 query | 开启 | WAM batch 开启 | AC-WM: `L_video`；WAM: `L_video + 5 L_CE` |

`cotrain` 默认以 `0.5` 概率抽 AC-WM micro-batch、`0.5` 概率抽 WAM
micro-batch。GigaPolicy-0.5 公开实现提供了 AC-WM/WAM mixed training 机制但没有公开
最终 sampling ratio，因此 `0.5` 是本项目明确记录的实验选择，不冒充论文常数。

与 GigaPolicy-0.5 原生连续机器人 action flow 的差异也固定如下：原模型对 14-D
连续 action 加噪并做 flow matching；本 ablation 为保持已有 GigaNav policy-only
可比性，继续使用 H=8 四类离散 CE。因而 WAM action slots 使用零 query，action
timestep 为 0；视觉流仍严格采用 GWP-0.5 的 `visual_flow_shift=2.0`：

```text
sigma = 2r / (1 + r)
Z_noisy = sigma * epsilon + (1 - sigma) * Z_future
V_target = epsilon - Z_future
t_visual = round(1000 * sigma)
```

### H=8 与 video chunk 的严格时间对齐

不能直接用整个 T4 chunk 的后三个 latent planes 作 future：一个 causal Wan T4
cache 对应 13 个 RGB frames，即 12 个 transition；而 GigaNav action horizon 是 8，
这样会留下 4 个没有 action condition 的 transition。也不把 H 改为 12，因为这会
破坏与已有 H=8 policy 实验的直接对比。

当前采用同一 T4 cache 的可复用切片：

```text
原缓存 RGB:       O(t), O(t+1), ..., O(t+12)
原缓存 latent:    Z[0], Z[1], Z[2], Z[3]
current/reference Z_ref    = Z[0]       # 1 RGB frame
future target     Z_future = Z[1:3]     # 8 RGB transitions
action target     A        = A[t:t+8]  # H=8
```

因此 video 输出为 `[B,16,2,56,112]`，覆盖当前帧之后的 8 帧；已有 T4 latent
无需重新 VAE 编码。旧 policy-only 复现仍可选 `legacy_policy_h8`，保留原来
`label_start=t+12` 的 delayed-label loader；新 video/cotrain 使用
`causal_h8_video`。二者不会被静默混合。

### Shared Wan block 内的因果关系

新 video/cotrain 路径安装无参数的 block-causal attention，仅替换已有 Wan
self-attention 的 K/V 可见关系，不改变参数名或官方 checkpoint 兼容性：

```text
主序列（Main sequence）:
  [ clean reference visual | zero state + action slots | noisy future visual ]

Q(reference) -> KV(reference)
Q(state/action) -> KV(reference, state/action)
Q(future) -> KV(reference, state/action, future)

Condition:
  UMT5 instruction [B,1,512,4096] -> 每个 Wan block 的 text cross-attention
```

这保证 policy/action hidden 看不到 GT future，不发生先验泄漏；video future 可以读到
action，因此 AC-WM 的 GT action 确实能够控制未来视觉。旧 policy-only 路径默认关闭
该补丁，保持已有 checkpoint 的计算图不变。

### 配置、脚本与初始化

```text
src/giga_nav/causal_attention.py             # 无参数 causal mask
src/giga_nav/model.py                        # shared video/policy forward + 两类 loss
src/giga_nav/data.py                         # legacy 与 causal-H8 两种 loader
scripts/train_giga_nav_multitask.py           # 统一 YAML 训练入口
scripts/run_giga_nav_multitask.sh             # 固定虚拟环境入口
config/giga_nav/giga_nav_wan21_h8_*.yaml      # 三种模式
```

正式 cotrain 从 `/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
重新加载 825 个 shape-compatible tensors，不继承已有 GigaNav policy checkpoint。
物理 BS=1、gradient accumulation=32、EBS=32、`lr=6e-5`、每 1000 optimizer
steps 保存。最初为避让 GPU0 的并行任务曾用 Adafactor 启动 39 steps 链路检查；
该 run 没有正式 checkpoint，已经停止且不进入实验比较。GPU1 释放后，正式 cotrain
改回与既有 GigaNav 一致的 AdamW，从原始 Wan 权重重新开始；模型、token、数据和
loss 均未改变。

完整模型/真实 batch 预检结果：官方 Wan 825 keys 加载成功；AC-WM 与 WAM 的
`video_velocity=[1,16,2,56,112]`，policy 输出 `action_logits=[1,8,4]`；联合反向
`L_video=0.21624`、`L_CE=2.15527`，Wan patch stem 与 policy head 均得到非零梯度，
峰值 allocated memory 23.53 GiB。该数值只证明完整数据流与梯度链路正确，不作为质量指标。

正式 AdamW run 使用 GPU1，tmux 为 `giga_nav_h8_cotrain_gpu1`，输出为
`log/giga_nav_wan21_h8_cotrain_adamw_gpu1_from_wan_20260907/`，TensorBoard tmux
为 `giga_nav_h8_cotrain_gpu1_tb`、端口 `6045`。此前 GPU0/Adafactor 的前 39 steps
只作为完整 mixed sampler 与梯度链路检查保留，不作为正式训练曲线。

## 闭环前期/中期方向跟随评测

修正后的闭环入口先排除 12-step bootstrap，再把剩余 expert action horizon 等分为
early、middle、late 三段，并分别记录：

- first-action 与 teacher action 的时间前缀准确率；
- 分段首尾 Navigation Error 的变化（goal progress）；
- agent 轨迹到 GT reference polyline 的平均欧氏距离；
- 实际发生位移的 step 与最近 GT 有向路径段之间的平均方向 cosine，以及同向比例；
- GT 路径与预测 early/middle/late/post-horizon 轨迹叠图。

这种口径优先回答模型在闭环误差尚未严重累积的前、中期是否跟随正确方向；后期因闭环
状态已偏离 teacher trajectory，逐时刻 action 一致率只作诊断，不要求始终保持。每条
轨迹写入 `result/giga_nav/<run>/trajectories/ep<ID>.{json,png}`。正式数值必须使用
已有 `step_005000.pt` 和滚动 13 帧协议重新测得；此前当前单帧协议结果不得混入。

完整链路单 episode codecheck 已通过。R2R train ep1 在排除 step0--11 bootstrap 后：
early 的 route-direction cosine 为 0.747、同向位移比例 75%，middle/late 的 cosine
分别约为 -0.009/-0.106；该 episode 最终 SR=0、Oracle Success=1，说明前段曾经过
成功邻域但没有正确 STOP，后续继续漂移。这只验证了新协议与“前期方向、后期变化”
分析链路，不作为 20-episode 聚合结论。
【已验证→`result/giga_nav/step5000_r2r_train_rolling13_codecheck2_20260905/summary.json`】

同一 `step_005000.pt` 的 R2R-train 前 20 条完整聚合结果如下：

| 指标 | 结果 |
| --- | ---: |
| SR / SPL | 0.0% / 0.0% |
| Oracle Success | 5.0%（1/20） |
| Navigation Error | 11.530 m |
| 平均步数 | 500 |
| STOP / FORWARD / LEFT / RIGHT | 0 / 9,491 / 102 / 407 |

| 对齐阶段 | Action agreement | Goal progress | Path deviation | Direction cosine | 同向位移比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early | 50.12% | -0.461 m | 2.278 m | -0.076 | 32.02% |
| Middle | 70.75% | -0.539 m | 2.986 m | 0.004 | 42.23% |
| Late | 49.74% | -0.596 m | 3.376 m | -0.203 | 24.68% |

20 条中，early 只有 8 条实现正 goal progress、7 条 direction cosine 为正；middle
分别为 6/8 条，late 为 2/4 条。模型 94.91% 的输出为 MOVE_FORWARD，且 10,000
步没有一次 STOP，因此 middle 70.75% action agreement 主要受 teacher action 的
前进多数类影响，不能解释为正确方向跟随。更直接的 goal progress 三段均为负，path
deviation 从 2.278 m 增至 3.376 m，说明聚合意义下前中期没有稳定沿目标路径推进，
后期继续恶化。唯一 Oracle Success episode 是 ep1，但未 STOP，最终仍以 8.793 m
Navigation Error 结束。
【已验证→`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】

## 验证状态与限制

已验证：

- 真实 Wan2.1-1.3B 权重加载（825 keys）；
- 真实 R2R latent、instruction embedding 和 action window 读取；
- 完整 Wan forward、action hidden readout、masked CE、反向传播和 optimizer update；
- 代码实测模型参数量及接口几何记录。
- checkpoint 恢复后 action-only inference 和 masked accuracy 入口通过一批真实窗口
  （`result/giga_nav/smoke_eval_step1.json`）；该随机/一步 checkpoint 的 66.67%
  只证明推理链路，不代表训练后导航指标。
- step5000 在 R2R-train 前 20 条 episode 上完成 first-action Habitat 闭环评测；
  SR/SPL/Oracle Success 均为 0，NE=10.519 m；该结果已标记为错误的当前单帧协议诊断。
- 闭环 server 会从 checkpoint 的原始 data config 自动识别 `+12`，采用滚动 13 帧
  输入；不会要求用户手工选择容易出错的模式。
- 已有 step5000 checkpoint 的 R2R-train 前 20 条滚动窗口评测完成；SR=0、Oracle
  Success=5%，三段平均 goal progress 均为负。

尚未验证：

- R2R-train 全量及 `val_unseen` 全量的滚动窗口闭环指标；
- Giga 原生连续 action flow 与离散 R2R head 的公平数值对比；
- 将真实 future visual latent 接回并进行 video/action co-training。

## 外部实现对应关系

- [GigaWorld-Policy 论文](https://arxiv.org/abs/2603.17240)：Giga 的共享 token
  排列、p=48 action horizon、视频/动作 flow matching 和 action-only 推理依据。
- [GigaWorld-Policy-0.5 model card](https://huggingface.co/open-gigaai/Giga-World-Policy-0.5)：
  Giga 0.5 的公开参数规模与 Wan2.2 5B/MoT 配置依据。
- [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1)：本地官方 Wan2.1
  DiT/VAE/UMT5 初始化来源。
