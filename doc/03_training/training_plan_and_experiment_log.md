# NAV 三阶段训练计划、课程与实验记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-010` |
| 类型 | 训练计划与实验记录总览 |
| 状态 | Live / Source of Truth |
| 更新时间 | 2026-08-28 |
| 职责 | 集中维护 V1 Stage One/Two/Three 训练变量、loss、数据配比、正式 run、历史 V0 实验和 streaming sample 语义。 |

## 当前入口结论

当前训练主线：Stage1 训练视频生成记忆与 action/text/interface 格式；Stage2 在同一视频生成范式中加入 3D/Pose hidden supervision；Stage3 加入 VLN policy/action loss，并与 Stage1/Stage2 loss 做比例混合以避免退化。

## 2026-08-28 两路 Stage3 延长至 step6000

GPU0 的 H4 full-history natural run 训练目标延长到绝对 optimizer step 6000。
它先完成原定 step2000；独立 tmux watcher 随后读取最新完整 checkpoint，
同时恢复 world model、policy head 和 AdamW state，并自动继续
`step2001..6000`。每 200 step 保存一次。GPU1 的 H1 short-history balanced
对照原本也配置了相同目标，但因其 natural eval 没有改善，已于 2026-08-28
按资源优先级停止，不再续到 step6000。

```text
watcher entry:
  scripts/watch_and_continue_v1_stage3_to_step6000.sh

GPU0 / H4 natural:
  current run: stage3_r2r_fullhistory_natural_a4_mb1_ebs16_from_stage2step3400_2k_20260828
  watcher tmux: nav_stage3_h4_to6000
  continuation TensorBoard metadata port: 6042

GPU1 / H1 balanced:
  current run: stage3_r2r_single_action_balanced_resume1400_to2000_gpu1_20260828
  status: stopped at in-memory step1473; latest reproducible checkpoint=step1400
  watcher: terminated
  TensorBoard 6041: terminated
```

Epoch 口径必须分开说明：H4 natural 有 35,941 个 unique windows，EBS16，
所以 1 epoch 约为 2,247 optimizer steps，6000 step 约为 2.67 epoch；若要求
数学上完整的 3.00 epoch，应训练到约 step6741。H1 balanced 使用四类有放回
cyclic sampling，没有严格的无放回 epoch；6000 step 对应 96,000 个样本曝光，
约等于其 88,602 个自然窗口总数的 1.08 倍，但不是 3 个 natural-data epoch。
本轮仅保留 H4，并以用户指定的绝对 step6000 为停止标准。

## 2026-08-28 Stage3 H4 自然分布协议

正式 Stage3 在 DEC-047 的 full episode prefix/Register 语义上，只替换动作目标与
采样分布：每个当前 `Z_obs` 预测连续 4 个 future primitive，形成
`action_class[B,4]`。输入 backbone 的 action slots 始终保持 Stage2 的 10 个；
最后一层得到 `hidden[B,10,1536]` 后只读取前 4 个，逐位置经共享 FP32 四分类
MLP 得到 `logits[B,4,4]`，后 6 个 hidden 不读出、不监督。这里没有额外的
`10→4` temporal projector，action loss 是前 4 个位置的普通 mean CE。

训练 window 使用原始自然分布：所有 unique full-prefix windows 每个 epoch shuffle
后无放回遍历，不做 TURN/STOP 复制、四类 cyclic balance 或 class weight。Stage2
visual/pose replay、`micro=1 × accumulation=16`、EBS16、step3400 初始化均保持不变。

运行状态：

```text
已停止（step 148）：
  stage3_r2r_fullhistory_mb1_ebs16_from_stage2step3400_2k_20260827

继续运行的历史短窗口对照：
  stage3_r2r_single_action_balanced_from_stage2step3400_2k_20260826

新协议正式任务（2026-08-28 02:09 已在 GPU0 启动）：
  scripts/run_v1_stage3_r2r_full_history_ebs16.sh
  --action-chunk 4
  --r2r-loader stage3_r2r_full_history_natural_action_chunk

run:
  log/v1_stage3_r2r_single_action/
    stage3_r2r_fullhistory_natural_a4_mb1_ebs16_from_stage2step3400_2k_20260828/

tmux:
  nav_stage3_fullhist_natural_a4
  nav_tb_stage3_fullhist

TensorBoard:
  port 6040
```

首个 optimizer step 已验证完整训练链路：140.14 s/step，CUDA max allocated
28.23 GiB；EBS16 共监督 64 个 action labels，`policy CE=1.4363`、
`visual replay=0.1043`、`pose replay=1.15e-4`、`grad norm=10.8439`，均为有限值。
optimizer state 建立后 GPU0 总占用约 31.4 GiB，未发生 OOM。

### 2026-08-28 H1 step1400 与 H4 step200 配对开环评测

旧短窗口 balanced H1 训练已在内存 step1418 停止；由于只按每 200 step 保存，
其最新可复现权重为 step1400。新 H4 full-history natural 训练继续运行，当前用于
对比的最新完整权重为 step200。

统一评测口径：R2R train 的 256 个同序 full-prefix natural windows，不做样本或
类别均衡，使用 correct instruction；batch size=4。两版均执行完整 Register +
shared Wan + policy head 前向。旧 H1 监督/评测 slot 0，新 H4 评测四槽并单列
slot 0，因此 slot-0 指标为严格 paired comparison。

| 模型 | checkpoint | 评测动作数 | CE | Accuracy | Macro Recall | Majority Baseline | 预测分布 STOP/MOVE/LEFT/RIGHT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 旧 balanced H1 | step1400 | 256 | 1.3362 | 21.48% | 34.02% | 73.44% | 70 / 38 / 96 / 52 |
| 新 natural H4，slot 0 | step200 | 256 | — | 73.44% | 25.00% | 73.44% | 0 / 256 / 0 / 0 |
| 新 natural H4，all slots | step200 | 1024 | 0.9603 | 66.21% | 25.00% | 66.21% | 0 / 1024 / 0 / 0 |

新 H4 的 raw accuracy 虽高，但预测全部为 MOVE_FORWARD，且 accuracy 与 majority
baseline 完全相等；STOP、TURN_LEFT、TURN_RIGHT recall 均为 0。当前 step200
属于明确的 majority collapse，不能解释为 policy 已学会 action。旧 H1 没有单类
塌缩并有一定 minority recall，但在 full-history natural 分布上总体 accuracy 很低。
H4 训练暂不停止；step400 及之后应复用相同 evaluator，重点检查 macro recall、
rare-action recall 和 prediction distribution 是否脱离全 MOVE。

```text
evaluator:
  scripts/eval_v1_stage3_action_chunk_compare.py

result:
  result/v1_stage3_action_chunk_compare/
    paired256_old1400_vs_h4step200_20260828/
      summary.json
      examples.json
      paired_first_slot_confusion.png
```

【已验证→result/v1_stage3_action_chunk_compare/paired256_old1400_vs_h4step200_20260828/summary.json】

#### 旧 H1 从 step800 到 step1400 是否真正改善

为排除上表 full-history protocol 与旧版 short-window 训练不一致的影响，进一步
使用旧模型原生 evaluator 做严格纵向对比：两次均为 short-window natural、
128 windows、correct instruction、seed=20260830。前 32 个落盘 example 的
sample ID 与 history length 完全一致。

| Checkpoint | Natural CE | Accuracy | Macro Recall | 预测分布 STOP/MOVE/LEFT/RIGHT |
| ---: | ---: | ---: | ---: | --- |
| step800 | 1.3601 | 15.63% | 26.80% | 13 / 7 / 108 / 0 |
| step1400 | 1.4696 | 14.84% | 26.55% | 11 / 7 / 110 / 0 |

结论：没有可测得的 natural policy 改善。训练日志中的 200-step balanced 平均
loss 从 step1–200 的 1.2891 降至 step1201–1400 的 1.1776，balanced accuracy
从 37.84% 升至 44.56%，说明优化器确实继续拟合了人工四类均衡训练目标；但
natural eval 的预测模式几乎不变，仍基本塌缩到 TURN_LEFT，且 CE 反而变差。
step800 之后的 loss 下降不能作为导航能力提升证据。

```text
step800 result:
  result/v1_stage3_single_action/
    modular_refactor_step800_natural_balanced128_seed20260830/summary.json

step1400 result:
  result/v1_stage3_single_action/
    step1400_natural128_seed20260830/summary.json
```

【已验证→result/v1_stage3_single_action/step1400_natural128_seed20260830/summary.json】

### 2026-08-28 H1 balanced 从 step1400 续训

为补足旧 H1 balanced 对照的训练步数，GPU1 从最后一个完整 checkpoint
`step_001400.pt` 续训到绝对 step 2000。恢复范围包括完整
world model、policy head 与 AdamW optimizer state；日志和新 checkpoint 使用
绝对 step `1401..2000`。原进程虽曾运行到内存 step1418，但未保存对应权重，
因此不可复现的 18 步不纳入续训起点。数据 sampler 使用新的 seed 重新实例化，
故这是严格的权重/优化器续训，但不是原数据序列的 bitwise continuation。

停止记录：该续训在内存 step1473 后由用户主动终止；step1600 尚未到达，因而
没有生成新的 checkpoint，最新可复现权重仍为原 `step_001400.pt`。训练进程、
step6000 watcher 与6041 TensorBoard 均已关闭，GPU1 已释放。停止原因是
step800→1400 的 natural eval 未改善，而 H4 full-history natural 更接近最终
Stage3 协议，故有限算力优先保留 H4。

```text
GPU: 1
tmux: nav_stage3_balanced_resume1400
TensorBoard: port 6041
run:
  log/v1_stage3_r2r_single_action/
    stage3_r2r_single_action_balanced_resume1400_to2000_gpu1_20260828/

source:
  log/v1_stage3_r2r_single_action/
    stage3_r2r_single_action_balanced_from_stage2step3400_2k_20260826/
      checkpoints/step_001400.pt

protocol:
  H=1 balanced-class cycle
  history K=1..7 micro chunks
  micro batch=1, gradient accumulation=16, EBS=16
  backbone_lr=2e-6, policy_lr=1e-4
  L = CE_4 + 0.25 * L_visual_replay + 0.05 * L_pose_replay
  save at absolute step 1600 / 1800 / 2000
```

绝对 step1401 已完成：144.59 s/optimizer step，CUDA max allocated 28.77 GiB；
`policy CE=1.1605`、balanced accuracy/macro recall 均为 37.5%，
`visual replay=0.0710`、`pose replay=0`，所有训练量均为有限值。该记录验证了
模型、policy head、AdamW state、完整 policy/replay forward-backward 的恢复链路。
【已验证→`log/v1_stage3_r2r_single_action/stage3_r2r_single_action_balanced_resume1400_to2000_gpu1_20260828/train.jsonl`】

## 2026-08-27 Stage3 全量历史 one-step 均衡对照（已停止）

结论：Stage3 导航样本不再将 Register history 人工截断为随机 `K=1..7` window。
对每一个被采样的单动作 target，history 必须是该 episode 从起点到当前
`Z_obs` 之前的完整 latent/action 前缀；原始历史仍不进入 Wan token stream，
只通过固定大小 Register recurrent update 压缩。

本次实现和运行口径：

```text
source checkpoint:
  log/v1_stage2_final_cotrain/
    stage2_from_step3000_continue_lr2e6_1k_20260825/
      checkpoints/step_003400.pt

R2R data:
  10,819 encoded/rendered train episodes
  35,941 full-prefix single-action targets
  history_micro = 1..14
  longest full prefix = 169 RGB frames
  invariant: start_micro=0, history_micro=obs_micro
  K=0 first-observation target: 暂不采样，与当前 Stage3 任务保持一致

model/loss:
  与正在运行的短窗口 Stage3 对照保持相同
  Single shared Wan backbone + Register + Z_obs + instruction + action query
  L = CE_4 + 0.25 * L_visual_replay + 0.05 * L_pose_replay
  backbone_lr=2e-6, policy_lr=1e-4
  四类 target 按 cyclic replacement 精确均衡

batch/runtime:
  GPU0
  physical micro batch=1
  gradient accumulation=16
  effective batch size=16
  steps=2,000, save_every=200
  TensorBoard port=6040
```

不同长度 prefix 只在 batch transport 时补零；`history_lengths` 控制每个样本
实际执行的 Register updates。模型分别滚动每个样本的 Register，使
`register_grad_tail=4` 仍以各自真实前缀末端为基准。该实现不跨 batch 保存场景状态，
但每个 target 的 Register
输入都包含完整 episode prefix，因此可与旧 `K=1..7` 截断窗口任务直接对照。

`micro=2 × accum=8` 完成第一个 optimizer step 后，AdamW state 已常驻显存；
第 2 步在 Stage2 replay backward 中申请额外 92 MiB 时 OOM。报错口径为 GPU0
总容量 47.40 GiB、进程占用 47.34 GiB、仅余约 40 MiB。因此 micro2 只能作为
显存校准失败记录，正式全量历史任务回退到 `micro=1 × accum=16`，EBS16 与其余
训练变量均不改变。

稳定配置首个完整 optimizer step 已验证：141.60 s/step，CUDA max allocated
28.23 GiB；四类 target 各 4 个，`policy CE=1.39149`、
`visual replay loss=0.10433`、`pose replay loss=0.000115`、
`grad norm=2.8113`，均为有限值。该记录只证明完整训练链路和资源配置有效，
不代表模型已收敛。【已验证→`log/v1_stage3_r2r_single_action/stage3_r2r_fullhistory_mb1_ebs16_from_stage2step3400_2k_20260827/train.jsonl`】

正式入口：

```text
scripts/run_v1_stage3_r2r_full_history_ebs16.sh
```

运行记录：

```text
log/v1_stage3_r2r_single_action/
  stage3_r2r_fullhistory_mb1_ebs16_from_stage2step3400_2k_20260827/
```

2026-08-21 更新：针对 `step_002000.pt` 的 pose 误差较大问题，新增
Stage2 frozen-backbone layer probe sweep。该实验不继续更新 DiT/Register，
而是冻结当前已训好的最终结构 checkpoint，使用 forward hook 抓取 Wan DiT
不同 block hidden state，分别训练 VGGT-style dense camera-query probe，从而
判断哪一层最具 3D/Pose 可读性。

代码入口：

```text
scripts/train_v1_stage2_probe_sweep.py
```

实验定义：

```text
checkpoint:
  log/v1_stage2_final_cotrain/
    stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/
      checkpoints/step_002000.pt

frozen:
  FinalStage2WanModel / Wan DiT backbone / Register / action interface / video branch

trainable:
  per-layer DenseCameraQueryPoseProbe

probe:
  current Z_obs hidden tokens [B,T,H,W,D]
    -> per-frame learned camera query cross-attention over spatial tokens
    -> MLP pose head [tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]

layers:
  4, 8, 12, 16, 20, 24, 29

data:
  RE10K only
  history_iw = 1
  T_latent = 4

metrics:
  pose_mse / translation_mae / rotation_angle_deg / fov_mae
```

本实验的判定逻辑：

```text
若某些中间层 probe 明显优于当前 final-prefix mean-pool pose head：
  说明当前 backbone 内部已有几何信息，主要问题是 readout 层与取层位置。

若所有层 probe 都不好：
  说明仅靠当前 visual+pose cotrain 尚未形成足够 3D 表征，需要增加
  depth / pointmap / correspondence 等 dense 3D supervision 或调整 Stage2 loss。
```

2026-08-18 起，后续正式训练必须按
`00_overview/final_formal_training_standard.md` 执行；不满足者只能记为
`gate / diagnostic / ablation`。下表是训练侧摘要：

```text
model/token:
  新版结构，保留多种 token 交互
  保留 video generation branch 与 policy branch
  显式维护两支路 attention / causal mask 关系
  保留 A_noise -> A_out policy/action path

action:
  A_hist       -> history action context，参与 Register update
  A_cur        -> video generation branch 的 current action condition
  A_noise/Aout -> Stage3 policy/action branch 输入与监督目标
  Register update 与 A_cur condition 均混入一定比例 all-zero empty action input

data:
  RE10K + SpatialVID + DL3DV 混合
  有 text 的样本直接编码 text
  无 text 的样本使用此前约定 empty text embedding

latent/history:
  T_latent = 4
  history length mix 对齐 InfiniteWorld 1 / 4 / 8 / 16 IW-chunk 等价长度

loss:
  Stage1: visual generation loss，并保留 action/text/token interface
  Stage2: visual + 3D/Pose mixed loss
  Stage3: policy/action loss + visual/3D replay，防止生成与几何能力退化

optimization:
  effective batch size = 16
```

昨日完成的 `formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124`
定位为：`fixed_rnull_unified + shared_tail_token` 子结构在 RE10K Stage2
formal gate 上通过，证明该子结构可收敛；它不是最终正式训练，因为它没有包含
RE10K/SpatialVID/DL3DV 混合、IW 1/4/8/16 历史长度混合、empty action
condition 比例、完整 A_noise policy branch 监督与 `ebs=16`。

## 2026-08-18 最终标准 Stage2 cotrain 正式入口

本节记录第一版严格绑定
`doc/00_overview/final_formal_training_standard.md` 的 Stage2 cotrain
正式训练代码与当前 run。该实现新建独立代码路径，避免与旧 V0/A/B、
legacy20 gate 和 earlier IWAligned gate 混淆。

代码入口：

```text
src/nav/v1/stage2_final/data.py
src/nav/v1/stage2_final/model.py
scripts/train_v1_stage2_final_cotrain.py
```

结构与数据边界：

```text
backbone:
  Infinite-World 当前 WanModel / Wan2.1-1.3B compatible DiT
  single shared WanBlock stream
  patch_embedding = 16ch；不使用 20-channel mask concat
  HPMC / latent_encoder removed

pre-backbone rollout:
  C_hist + A_hist -> Register R_t

main token stream:
  Z_obs
  A_noise
  Z_future_noise

condition:
  Register R_t
  text / empty text
  A_cur

target/loss-only:
  Z_future_target
  pose_target / pose_mask
```

当前 run 选择 `Register R_t` 走 `condition_memory`，而不是 `main_prefix`。
原因是 `main_prefix` 在 IW16 history + full Stage2 cotrain 下会把 visual
prefix token 数显著抬高，实测在 48G 单卡上 OOM；`condition_memory` 是最终
标准允许的两种 Register 注入形式之一，同时保持 `Z_obs` 在 main token stream。

长历史训练采用正式的 truncated BPTT：

```text
C_hist 全部参与 Register rollout；
早期 history update 在 no_grad 下只贡献 Register state；
最后 register_grad_tail=4 个 micro history update 保留梯度；
DiT backbone / condition projection / pose head / action-token path 正常反传。
```

这不是把 history 截短；它只是避免 IW16 对应 107 个 T=4 micro chunks 时，
单个样本的反向图随 history 长度线性爆炸。

数据与采样：

| 项 | 设置 |
| --- | --- |
| manifest | `/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl` |
| latent root | `/sharedata/NAV/derived/v1/t4_micro_latents_spatial20` |
| 数据混合 | `re10k=0.20, spatialvid=0.45, dl3dv=0.35` |
| history mix | IW `1 / 4 / 8 / 16` |
| micro history | IW1=`7`, IW4=`27`, IW8=`54`, IW16=`107` |
| empty action | `empty_hist_prob=0.10`, `empty_cur_prob=0.10` |
| effective batch size | `16 = micro_bs 1 × grad_accum 16` |
| checkpoint init | `/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` |
| Wan keys loaded | `825` shape-compatible keys |
| Stage2 loss | `L = L_visual + 0.1 * L_pose` |

可用训练 windows：

| history | windows |
| --- | ---: |
| IW1 | 206,193 |
| IW4 | 121,654 |
| IW8 | 65,305 |
| IW16 | 35,654 |

当前正式 run：

| 项 | 内容 |
| --- | --- |
| run | `stage2_final_cotrain_condmem_iw14816_tbptt4_20260818_113721` |
| run dir | `log/v1_stage2_final_cotrain/stage2_final_cotrain_condmem_iw14816_tbptt4_20260818_113721/` |
| train tmux | `nav_stage2_final_cotrain` |
| TensorBoard tmux | `nav_stage2_final_tb` |
| TensorBoard port | `6017` |
| steps | `2000` |
| save_every | `200` |
| dtype | `bf16` |
| optimizer | `AdamW(lr=1e-5, weight_decay=0.01)` |

启动命令：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 \
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python \
  scripts/train_v1_stage2_final_cotrain.py \
  --run-name stage2_final_cotrain_condmem_iw14816_tbptt4_20260818_113721 \
  --device cuda:0 \
  --register-injection condition_memory \
  --register-grad-tail 4 \
  --steps 2000 \
  --batch-size 1 \
  --grad-accum 16 \
  --save-every 200 \
  --log-every 1 \
  --tensorboard-port 6017
```

首个 optimizer step：

| metric | value |
| --- | ---: |
| `train/loss` | 0.9042 |
| `train/loss_visual` | 0.9012 |
| `train/loss_pose` | 0.0304 |
| sampled history | IW16 / 107 micro |
| sampled dataset | DL3DV |
| seconds/opt step | 69.83 |
| max allocated memory | 28.05GiB |

启动过程中的重要失败与修复：

| 尝试 | 现象 | 结论/处理 |
| --- | --- | --- |
| reentrant checkpoint | backward graph error | 改为 Wan non-reentrant checkpoint，沿用旧 Stage1 稳定策略 |
| `main_prefix` Register | 48G 单卡 OOM | 当前正式 run 改用标准允许的 `condition_memory` |
| full Register BPTT | IW16 history OOM | 保留完整 Register state，采用 `register_grad_tail=4` truncated BPTT |

Step400 intermediate eval（2026-08-18）：

| 项 | mixed visual eval | RE10K pose eval |
| --- | --- | --- |
| checkpoint | `step_000400.pt` | `step_000400.pt` |
| output | `result/v1_stage2_final_cotrain/step400_mixed64_decode4_20260818_222425/` | `result/v1_stage2_final_cotrain/step400_re10k_pose64_decode4_20260818_222425/` |
| batches | 64 | 64 |
| data | `re10k=0.20, spatialvid=0.45, dl3dv=0.35` | `re10k=1.0` |
| history | IW `1/4/8/16` | IW `1` |
| decoded samples | 4 | 4 |
| `eval/loss_visual_velocity_mse` | 0.1296 | 0.1444 |
| `eval/latent_x0_mse` | 0.0540 | 0.0641 |
| `eval/baseline_noisy_latent_mse` | 0.9758 | 1.0661 |
| `eval/latent_mse_improvement_vs_noisy` | 0.9218 | 1.0020 |
| `eval/latent_x0_cosine` | 0.9381 | 0.9446 |
| `eval/loss_pose_mse` | 0.00044 overall；RE10K 子集 0.00699 | 0.00824 |
| `eval/pose_translation_mae` | 0.0046 overall | 0.0685 |
| `eval/pose_rotation_angle_deg` | 0.95 overall | 12.93 |
| `eval/pose_fov_mae` | 0.0042 overall | 0.0822 |
| RGB PSNR | 25.37 dB | 27.15 dB |

解释：

- mixed eval 的 pose 指标被大量无 pose 样本稀释，因此 3D 结论主要看
  RE10K-only eval。
- 生成指标使用 one-step RFlow x0 proxy：`x0 = z_noisy - t * v_pred`，
  不是完整多步 sampler。
- 8 个导出的 `pred_future_proxy.mp4` 均可打开，均为 13 帧，像素 mean/std
  正常且存在帧间变化；没有复现早期黑屏/坏视频问题。

Step400 full denoise sampler（2026-08-19）：

为检查真实生成效果，新增完整多步 RFlow sampler：

```text
scripts/sample_v1_stage2_final_denoise.py
```

采样方式：

```text
z_1 ~ N(0,I)
t: 1 -> 0, shifted RFlow, 30 denoise steps
每步 DiT 预测 reversed velocity: noise - x0
更新: z <- z - dt * v_pred
```

运行：

```text
result/v1_stage2_final_denoise/step400_mixed4_full30_20260819_003649/
```

4 个 mixed 样本对比：

| sample | dataset/history | one-step proxy PSNR | full 30-step PSNR |
| --- | --- | ---: | ---: |
| 000 | SpatialVID / IW4 | 20.44 | 19.46 |
| 001 | SpatialVID / IW8 | 37.86 | 25.02 |
| 002 | SpatialVID / IW8 | 25.78 | 21.10 |
| 003 | DL3DV / IW4 | 17.38 | 16.76 |

均值：

| metric | one-step proxy | full 30-step denoise |
| --- | ---: | ---: |
| RGB PSNR | 25.36 dB | 20.59 dB |
| RGB L1 | 0.0485 | 0.0655 |
| latent MSE | 0.0274 | 0.0632 |

解释：

- one-step proxy 的 `z_noisy=(1-t)z_target+t noise` 含有 GT target 成分，
  适合做 denoise proxy 指标，但不是纯生成。
- full denoise 从纯噪声开始，更接近真实 inference，当前 step400 下质量明显
  更不稳定，尤其高难度样本会退化。
- 单次 DiT forward 约 `0.96–1.13s`；30-step denoise 每样本 DiT 时间约
  `28.7–33.9s`，4 样本总 wall-clock 约 `170s`，包含 checkpoint load、
  VAE decode 与视频写盘。

结论：step400 已具备可用 one-step denoise 表征，但完整采样质量尚未稳定；
后续应继续观察 step600/800，并考虑增加 sampler-consistency / high-t
训练权重或在 eval 中固定 timestep 分桶分析。

2026-08-17 的最新结论是：此前 `V1SharedWanBackbone` Stage2 版本只加载约 242 个 Wan shape-compatible key，视频支路与旧可收敛 StageOne 差异过大；现已新增 `IWAlignedWorldNavModel`，把 video generation branch 回贴到 InfiniteWorld/WanModel 原生 `patch_embedding / time_embedding / Wan blocks / denoise head` 路径，并恢复 825 个官方 Wan key 加载。旧 StageOne legacy20 路径已精确复现，step1/step10/step20 loss 分别约 `1.613/0.815/0.572`；本轮又确认 action-tail RoPE/checkpoint 修复后旧 no-action-tail step1 仍完全等于 `1.6130983456969261`。Stage2 当前采用 `strict video pass + separate current-clean pose-probe pass`：video loss 不再因 pose supervision 改变主生成 token layout，pose 另用共享 WanBlock 的 current-clean prefix hidden 监督。该 two-pass graft 已用完整模型 1-step 验证，`loss_visual=1.5098`、`loss_pose=0.5234`，峰值约 `27.97GiB`。下一步的关键 gate 是多步训练观察 video loss 是否重新接近 legacy20 旧曲线，同时 pose loss 稳定下降。

## 2026-08-17 IW/Wan 对齐回贴实验

### 旧版 20-channel InfiniteWorld + Register 复现

旧版有效结构并不等于当前 `Infinite-World/infworld/models/dit_model.py` 工作树。
真正对应 2026-08-11/12 快速收敛 run 的结构是：

```text
InfiniteWorld/Wan 20-channel branch:
  latent channels: 16
  condition mask channels: 4
  patch_embedding.weight: [1536, 20, 1, 2, 2]

NAV adapter:
  HPMC / latent_encoder removed
  SpatialRegisterMemory: [B,16,4,H,W]
  StageOneActionInterface legacy action probe: 820,516 params
  no token_type_embedding
  no separate shared_action checkpoint group
```

为避免和当前 V1/IW 改造互相覆盖，已建立独立 worktree：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20
```

运行旧 StageOne 复现必须设置：

```bash
export NAV_INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20
export PYTHONPATH=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/src:$NAV_INF_WORLD_ROOT:$PYTHONPATH
```

`scripts/run_v1_wan_stage1_t4_history.sh` 已默认指向该 legacy20 worktree。

结构审计与历史 run 对齐：

| 项 | 历史 run | 2026-08-17 legacy20 复现 |
| --- | ---: | ---: |
| trainable params | 1,422,300,004 | 1,422,300,004 |
| backbone params | 1,421,390,400 | 1,421,390,400 |
| register params | 89,088 | 89,088 |
| action interface params | 820,516 | 820,516 |
| patch embedding | `[1536,20,1,2,2]` | `[1536,20,1,2,2]` |
| Wan loaded keys | 825 + partial patch expand | 825 + partial patch expand |

收敛诊断：

| run | 结构 | step1 | step10 | step20 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `log/v1-stageone-text-fullmix-wan21official-lp-mb1-ebs16-5000-20260812-020056/` | 历史真实 run | 1.386 | 0.712 | 0.507 | 旧版目标曲线 |
| `log/legacy_stage1_t4_fullmix_convergence_20260817_014255/` | 错误：旧 NAV 脚本 + 当前 16ch/token-type IW | 1.210 | 未达，step9=1.018 | - | 未复现旧结构 |
| `log/legacy20_stage1_t4_fullmix_convergence_20260817_021207/` | 正确：legacy20 20ch IW + legacy action interface | 1.613 | 0.815 | 0.572 | 基本复现旧版快速下降 |
| `log/legacy20_stage1_t4_fullmix_convergence_20260817_024822/` | 同一旧版命令复跑；ebs16；`history_iw_chunks=1,4,8,16` | 1.613 | 0.815 | - | step1–10 与 `021207` 几乎逐点重合，确认旧命令可稳定复现 |
| `log/legacy20_ropeinfer_noaction_regression_20260817_050052/` | action-tail RoPE/checkpoint 修复后的旧 no-action-tail 防回归；ebs16 | 1.613 | - | - | step1 与 `021207` 完全一致，确认修复对旧生成路径 no-op |

`20260817_024822` 的前 10 个 optimizer step 明细：

| step | loss | history IW chunks | history micro steps |
| ---: | ---: | ---: | ---: |
| 1 | 1.6131 | 16 | 107 |
| 2 | 1.6934 | 16 | 107 |
| 3 | 1.2228 | 1 | 7 |
| 4 | 1.1413 | 4 | 27 |
| 5 | 0.8754 | 16 | 107 |
| 6 | 0.9833 | 8 | 54 |
| 7 | 0.8911 | 8 | 54 |
| 8 | 0.9465 | 1 | 7 |
| 9 | 0.8499 | 16 | 107 |
| 10 | 0.8151 | 16 | 107 |

资源观测：`batch_size=1`、`gradient_accumulation_steps=16`、全参训练，峰值
`peak_reserved≈37.3GiB`。wall-clock 会随抽到的 history 长度显著波动；16 IW chunks
对应 107 个 T=4 micro chunks，是主要慢项。

因此，后续“在旧版有效生成支路上加最终设定”必须从
`legacy20_stage1_t4_fullmix_convergence_20260817_021207` 对应结构出发，而不是
当前 16ch/token-type IW branch。

### legacy20 Stage2 two-pass pose bridge

目标：在不改变旧版可收敛 video generation branch 的前提下，把 Stage2 的
current-chunk pose supervision 接到同一个 Wan/DiT backbone 上。

核心做法：

```text
video pass:
  input = old legacy20 Register + local memory + noisy future
  loss  = RFlow visual loss
  注意：不拼接 z_obs/current chunk 到 video prefix，避免破坏旧生成支路。

pose pass:
  input prefix = clean z_obs/current chunk + local memory
  dummy future = zeros_like(z_obs), t=0，仅用于满足 Wan block 非空 noisy segment 假设
  readout = prefix_video_hidden 的 z_obs token
  loss    = FramePoseHead(hidden) -> per-frame pose MSE
```

关键修复：

- `scripts/train_v1_stage2_legacy20_re10k_pose_video.py`
  - 删除会污染 video branch 的 `current_latent_prefix=z_obs`。
  - video pass 严格保持 legacy20 `Register + local_latent -> future` 路径。
  - pose pass 单独 forward，读取 clean-prefix current hidden。
  - 因 IW/Wan block 不支持空 noisy segment，pose pass 使用 `dummy_future=zeros_like(z_obs)`，
    但 pose loss 不从 dummy segment 读取。

1-step sanity：

| run | 设置 | loss_visual | loss_pose | peak reserved |
| --- | --- | ---: | ---: | ---: |
| `log/v1_stage2_legacy20_re10k_pose_video/debug_legacy20_stage2_2pass_dummyfuture_1step_20260817_030400/` | RE10K-only；`λ_pose=0.1`；two-pass | 2.0678 | 0.6238 | 28.5GiB |

20-step gate：同一 seed、同一数据顺序、同一模型初始化，对比 `λ_pose=0`
和 `λ_pose=0.1`。该 gate 的目的不是证明最终泛化，而是确认新增 pose loss
不会立即破坏 visual branch 的快速下降。

| step | visual-only `λ=0` visual | pose-cotrain `λ=0.1` visual | pose-cotrain pose |
| ---: | ---: | ---: | ---: |
| 1 | 2.4826 | 2.4826 | 0.6262 |
| 2 | 0.5853 | 0.5848 | 0.5427 |
| 3 | 0.7251 | 0.7227 | 0.4506 |
| 5 | 0.5421 | 0.5440 | 0.3607 |
| 7 | 0.4470 | 0.4488 | 0.2398 |
| 10 | 0.4742 | 0.4721 | 0.1174 |
| 15 | 0.3855 | 0.3882 | 0.0183 |
| 20 | 0.3510 | 0.3532 | 0.0023 |

统计：

| run | first5 visual mean | last5 visual mean | first5 pose mean | last5 pose mean | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cmp_legacy20_stage2_lambda0_video_gate_20step_20260817_030453` | 0.9906 | 0.4146 | 0.6038 | 0.5621 | 31.0GiB |
| `cmp_legacy20_stage2_lambda0p1_pose_gate_20step_20260817_030531` | 0.9903 | 0.4174 | 0.4836 | 0.0100 | 31.0GiB |

结论：

- `λ_pose=0.1` 下 visual 曲线与 `λ_pose=0` 基本重合，说明 two-pass pose
  supervision 没有像旧 `current_latent_prefix=z_obs` 方案那样伤害生成支路。
- pose train loss 快速下降，说明 current hidden -> pose head 的梯度链路有效。
- 该结果作为 bridge gate 通过；最终 Stage2 仍需要更长训练和视频生成样本/
  VBench-style 质量检查，不能仅凭 20-step train/eval 宣称完整完成。

保存 checkpoint 后的 held-out eval：

| 项 | 内容 |
| --- | --- |
| train run | `log/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_save20_20260817_031122/` |
| checkpoint | `step-000020.pt` |
| eval run | `result/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_save20_eval16_20260817_031439/` |
| eval script | `scripts/eval_v1_stage2_legacy20_re10k_pose_video.py` |
| eval setting | RE10K held-out seed `20260834`，`max_batches=16` |

| metric | value |
| --- | ---: |
| `eval/loss_visual_velocity_mse` | 0.4016 |
| `eval/latent_x0_mse` | 0.4103 |
| `eval/baseline_noisy_latent_mse` | 1.1025 |
| `eval/latent_mse_improvement_vs_noisy` | 0.6921 |
| `eval/loss_pose_mse` | 0.0051 |
| `eval/pose_translation_mae` | 0.0414 |
| `eval/pose_rotation_angle_deg` | 6.7475 |
| `eval/pose_fov_mae` | 0.0891 |

解释：20-step checkpoint 在 held-out seed 上已经同时保留 visual denoise 能力并读出
pose；这证明 two-pass Stage2 的完整模型/数据/梯度链路成立。它仍只是短程验证，
下一步进入正式 Stage2 时需要拉长 steps、扩大 RE10K 样本、增加生成样本检查。

100-step 稳定性验证：

| 项 | 内容 |
| --- | --- |
| train run | `log/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_100step_20260817_031657/` |
| checkpoints | `step-000050.pt`, `step-000100.pt` |
| data | RE10K-only，`max_train_samples=256`，`history_steps=1` |
| loss | `L = L_visual + 0.1 * L_pose` |
| peak reserved | about 31.0GiB |

训练分段均值：

| steps | visual mean | pose mean |
| --- | ---: | ---: |
| 1–20 | 0.5456 | 0.1817 |
| 21–50 | 0.3502 | 0.0164 |
| 51–80 | 0.2083 | 0.0054 |
| 81–100 | 0.1850 | 0.0045 |
| last10 | 0.2086 | 0.0036 |

关键 step：

| step | visual | pose |
| ---: | ---: | ---: |
| 1 | 1.6995 | 0.5901 |
| 10 | 0.3808 | 0.1381 |
| 20 | 0.3120 | 0.0028 |
| 50 | 0.3851 | 0.0108 |
| 75 | 0.1768 | 0.0041 |
| 100 | 0.2860 | 0.0016 |

100-step held-out eval：

| 项 | 内容 |
| --- | --- |
| checkpoint | `log/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_100step_20260817_031657/step-000100.pt` |
| eval run | `result/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_100step_eval32_20260817_033039/` |
| eval setting | held-out seed `20260835`，`max_batches=32` |

| metric | value |
| --- | ---: |
| `eval/loss_visual_velocity_mse` | 0.1691 |
| `eval/latent_x0_mse` | 0.2520 |
| `eval/baseline_noisy_latent_mse` | 1.0011 |
| `eval/latent_mse_improvement_vs_noisy` | 0.7490 |
| `eval/loss_pose_mse` | 0.0038 |
| `eval/pose_translation_mae` | 0.0477 |
| `eval/pose_rotation_angle_deg` | 6.9171 |
| `eval/pose_fov_mae` | 0.0741 |

100-step 结论：two-pass Stage2 在更长训练上同时推动 visual denoise 与 pose readout
收敛。相较 20-step checkpoint，held-out visual velocity MSE 从 0.4016 降到
0.1691，pose MSE 维持在 0.0038。下一步必须补视频样本导出/解码检查，避免只用
latent/RFlow proxy 指标判断生成质量。

100-step decode/sample export：

| 项 | 内容 |
| --- | --- |
| eval+decode run | `result/v1_stage2_legacy20_re10k_pose_video/legacy20_stage2_posebridge_100step_decode2_20260817_033525/` |
| command support | `scripts/eval_v1_stage2_legacy20_re10k_pose_video.py --decode-rgb --save-videos` |
| saved samples | `samples/*/{obs.mp4,gt_future.mp4,pred_future_proxy.mp4}` |
| decode batches | 2 |
| note | `pred_future_proxy` 是 one-step RFlow x0 proxy，不是完整 30-step sampler；用于检查生成支路是否非黑、可解码、统计合理。 |

decode metrics：

| metric | value |
| --- | ---: |
| `eval_decode/rgb_mse` | 0.0135 |
| `eval_decode/rgb_l1` | 0.0789 |
| `eval_decode/rgb_psnr_db` | 18.6934 |
| `eval_decode/rgb_pred_mean` | 0.4708 |
| `eval_decode/rgb_pred_std` | 0.2359 |
| `eval_decode/rgb_target_mean` | 0.4772 |
| `eval_decode/rgb_target_std` | 0.2414 |

视频可播放/非黑帧抽检：

| sample | file | frames | mean | std | min/max |
| --- | --- | ---: | ---: | ---: | --- |
| `000_re10k__009f6e6c5aea0441_000000_start5` | `pred_future_proxy.mp4` | 13 | 105.39 | 50.55 | 0 / 255 |
| `001_re10k__001511b4a282e504_000000_resampled_start1` | `pred_future_proxy.mp4` | 13 | 132.40 | 69.78 | 0 / 255 |

该检查补上了最基本的生成支路可视化 gate：视频可由 VAE 解码、可播放、像素统计与
GT/obs 同量级。下一步若要作为正式论文实验，还需要完整 sampler 下的生成视频、
更大 held-out set，以及与旧 StageOne / InfiniteWorld baseline 的同协议比较。

### legacy20 与正式 action/policy 设定的 gap audit

当前已经验证成立的是：

```text
legacy20 generation branch:
  20-channel mask patch stem
  Register prefix + local memory
  Wan/InfiniteWorld 原生 hist/noisy split
  RFlow video loss

legacy20 Stage2 bridge:
  video pass 完全保持 generation branch
  pose pass 读 clean current prefix hidden
```

但这还没有等价于正式 V1 的全部 action/policy 设定。差距如下：

| 正式不变量 | legacy20 当前状态 | 风险/说明 |
| --- | --- | --- |
| `A_hist` 作为独立 action tokens 参与 RegisterCell update，禁止 bias | 当前 `StageOneActionInterface.condition_history_chunk` 仍是 additive latent bias | 为了复现旧版快速收敛保留；与正式 no-bias action invariant 不一致 |
| `A_cur` 作为 DiT condition/context，只给 future visual tokens 读 | legacy20 仍使用 InfiniteWorld 原生 `action_encoder(move, view)` additive path | 能保留旧生成支路，但不是正式 condition token 方案 |
| `A_noise -> A_out` 进入 single shared WanBlock token stream | 2026-08-17 已给 legacy20 WanModel 增加可选 `shared_action_tokens` tail；action tail 经过同一组 Wan blocks，head 前剥离；旧 `action_query/probe` 仍仅兼容旧 checkpoint | 已从“不在 backbone 内”推进到“legacy20-compatible shared-block graft”；下一步要把 learnable query 改成真正 `A_noise(action_t)` 和 action-flow decoder |
| policy-safe attention：action tokens 不得 attend future | action-tail 路径已用 SDPA mask 禁止 action rows 读取 future noisy visual keys；不传 action tail 时旧 flash-attn 路径保持不变 | 当前 mask 只覆盖 action-tail graft；完整 16ch formal V1 仍需独立验证 |
| V1 正式删除 20-channel mask，回到 16ch patch stem | legacy20 有效收敛依赖 20-channel mask branch | 直接删除会回到此前不收敛风险；必须单独重新过 generation gate |

因此当前状态应定义为：

```text
已完成:
  旧版 IW/Register generation 复现；
  在不伤旧 generation branch 的前提下接入 Stage2 pose supervision；
  训练/eval/视频导出证明 generation + 3D 可以同时收敛；
  legacy20 Wan blocks 已支持可选 policy-safe action tail，且旧路径防回归通过。

未完成:
  A_hist no-bias token 化的多步收敛 gate；
  Stage3 policy/action-flow 从 smoke latent cache 扩展到 full R2R obs latent cache；
  16ch/no-20-channel-mask formal V1 重新通过 generation convergence gate。
```

action-tail graft 验证（2026-08-17）：

| 项 | 结果 |
| --- | --- |
| 代码 | `Infinite-World-legacy20/infworld/models/dit_model.py`，`src/nav/v1/models/iw_aligned.py` |
| full-model 1-step | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_actiontail_1step_20260817_041636/` |
| 初始化 | Wan2.1 official safetensors；loaded keys=825；patch stem 16ch→20ch partial expand |
| 数据 | RE10K formal Stage2 sample；latent `[16,4,56,112]`；policy/action horizon=10 |
| step1 | `loss_visual=1.0371`，`loss_pose=0.8398`，max memory `39.74GiB` |
| 防回归 | `log/legacy20_stage1_no_actiontail_regression_20260817_041744/`；旧路径 step1 loss `1.6130983456969261`，与复现旧 run 完全一致 |

关键实现细节：

```text
shared_action_tokens present:
  visual tokens keep original Wan 3D RoPE
  action tail skips RoPE because it has no spatial grid
  action query rows cannot attend future noisy visual keys
  video head receives only visual tokens

A_cur video condition:
  policy H=10 cannot be directly fed to IW ActionEncoder
  before native video-action encoder:
    H=10 action chunk -> left-pad/crop to 81-frame IW action sequence
  shared action tail still keeps H=10
```

后续路线：

1. **Legacy20 graft route**：继续保持 legacy20 generation branch 为主线，把
   action tail 从 learnable query 改为真正 `A_noise(action_t)`，增加
   action-flow decoder，并用 R2R rendered-policy H=10 数据验证 Stage3 loss 与
   policy-only 推理速度。
2. **Formal 16ch route**：回到 16ch shared token stream，实现完全 no-mask-channel、
   `A_hist/A_cur/A_noise` 三类 token 和 action-flow decoder；但必须重新通过旧版
   generation 收敛 gate，否则不能替代 legacy20 主线。

### legacy20 graft: Stage3 `A_noise -> A_out` action-flow 链路

2026-08-17 已将 DEC-048 的 shared action tail 从 learnable query fallback 升级为
正式 `A_noise(action_t)` 输入：

```text
A_noise [B,H_nav=10,6]
  + action_timestep
  + action position embedding
  -> shared legacy20 Wan blocks action tail
  -> action_velocity_head / primitive_head

L_stage3_policy =
  MSE(action_velocity, action_target - A_noise)
  + lambda_ce * CE(primitive_logits, primitive_id)
```

代码：

```text
src/nav/v1/models/iw_aligned.py
scripts/train_v1_stage3_iw_aligned_r2r_policy.py
```

Stage3 真数据链路验证：

| 项 | 结果 |
| --- | --- |
| run | `log/v1_stage3_iw_aligned_r2r_policy/stage3_actionflow_r2r_smokelatent_1step_20260817_042532/` |
| 数据 | rendered-policy H=10 manifest + R2R rendered obs Wan-VAE latent smoke cache |
| 样本形状 | `history_latents=[4,16,1,56,112]`，`z_obs=[16,1,56,112]`，`a_noise/action_target=[10,6]` |
| 初始化 | Wan2.1 official safetensors，loaded keys=825，20ch partial patch expand |
| step1 | `loss_action_flow=1.3203`，`CE_aux=2.6719`，`total=1.5859` |
| 速度/显存 | `3.97s/step`，`28.07GiB` |

Stage2 兼容性验证（改为 A_noise 后）：

| 项 | 结果 |
| --- | --- |
| run | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_actionnoise_stage2_1step_20260817_042700/` |
| step1 train | `loss_visual=1.8560`，`loss_pose=0.4824` |
| step1 eval | `loss_visual=1.1076`，`loss_pose=0.4683` |

Stage2 two-pass pose-probe 更新（2026-08-17）：

| 项 | 结果 |
| --- | --- |
| run | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_poseprobe_1step_20260817_045819/` |
| video pass | strict legacy/IW layout：`image_cond=Register(history,A_hist)`，`local_memory=z_obs last latent frame`，`x=z_future_noisy` |
| pose pass | separate current-clean probe：`image_cond=z_obs`，`local_memory=z_obs last latent frame`，`x=zeros_like(z_obs)`，读 current clean-prefix hidden |
| action tail | `A_noise(action_t)` H=10 进入 shared WanBlock tail；self-attention 由当前 `grid_sizes` 推断 visual/action split，checkpoint backward 不再受 mutable attribute 覆盖 |
| step1 train | `loss_visual=1.5098`，`loss_pose=0.5234` |
| step1 eval | `loss_visual=1.2806`，`loss_pose=0.5127` |
| 速度/显存 | `17.72s/step` train window，max memory `27.97GiB` |
| 防回归 | `log/legacy20_ropeinfer_noaction_regression_20260817_050052/`：旧 no-action-tail Stage1 step1 `1.6130983456969261`，与旧复现完全一致 |

Stage2 two-pass 30-step generation-preservation gate：

| 对照 | run | λ_pose | step30 train visual | step30 eval visual | step30 eval pose | 显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cotrain | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_cotrain_gate30_20260817_050925/` | 0.1 | 0.3158 | 0.5386 | 0.0999 | 29.26GiB |
| video-only | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_strict_videoonly_gate30_20260817_050925/` | 0.0 | 0.3146 | 0.5382 | - | 27.94GiB |

配对 visual loss 统计：

```text
common train steps = 30
mean(cotrain_visual - videoonly_visual) = +0.00034
max_abs_diff = 0.00608

eval/loss_visual:
  step10: 0.69540 / 0.69528
  step20: 0.60807 / 0.60932
  step30: 0.53856 / 0.53823

cotrain eval/loss_pose:
  step10: 0.3496
  step20: 0.2148
  step30: 0.0999
```

结论：当前 two-pass Stage2 结构已经通过短程 gate：加入 pose supervision 后，
video generation loss 与同结构 video-only 基本逐步重合，同时 pose loss 明显下降。
这证明问题的关键不再是“3D 监督改变了生成支路 layout”，下一步应转向更长步数、
更大 effective batch、更多数据，以及后续 Stage3 full-cache policy/replay。

可复用分析脚本：

```bash
python NAV/scripts/analyze_stage2_gate.py \
  --cotrain NAV/log/v1_stage2_iw_aligned_re10k_pose_video/<cotrain_run> \
  --videoonly NAV/log/v1_stage2_iw_aligned_re10k_pose_video/<videoonly_run> \
  --json-out NAV/result/stage2_gate/<gate_name>/report.json
```

30-step gate 的标准报告已保存：

```text
NAV/result/stage2_gate/iw_aligned_gate30_20260817_050925/report.json
```

Stage2 two-pass 300-step generation-preservation gate 已启动：

| 对照 | run | GPU | steps | λ_pose | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| cotrain | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_cotrain_gate300_20260817_052230/` | 0 | 300 | 0.1 | Running |
| video-only | `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_strict_videoonly_gate300_20260817_052230/` | 1 | 300 | 0.0 | Running |

启动 sanity：

```text
step1 visual:
  cotrain/video-only = 1.406498 / 1.406498

video-only step5 visual = 0.705880
```

step50 partial report：

```text
report:
  NAV/result/stage2_gate/iw_aligned_gate300_20260817_052230/report_partial_step50.json

train visual paired common steps:
  common logged steps = 11
  mean(cotrain_visual - videoonly_visual) = +0.00134
  max_abs_diff = 0.00371

eval/loss_visual @ step50:
  cotrain/video-only = 0.43922 / 0.43414

cotrain eval/loss_pose @ step50:
  0.03918

speed after pausing concurrent R2R obs-latent encoders:
  cotrain ≈ 11.0 s/step
  video-only ≈ 7.5 s/step
```

中期结论：到 step50 为止，cotrain 的 video loss 仍与同结构 video-only
高度一致，pose eval 已从初始约 0.4 量级降到 0.039。该 gate 继续运行到
step300；期间 R2R obs-latent encoder 被临时 `SIGSTOP`，完成 gate 后需要
`SIGCONT` 恢复。

自动收尾 watcher：

```text
tmux session:
  nav_stage2_gate300_watch_20260817

log:
  NAV/log/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_gate300_watch_20260817_052230.log

behavior:
  wait for cotrain/video-only training PIDs to exit
  run scripts/analyze_stage2_gate.py
  write final report:
    NAV/result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json
  resume paused R2R obs-latent encoders with SIGCONT
```

2026-08-17 06:03 更新：由于 GPU1 在 video-only gate 结束后接着运行
legacy20 cmdcheck，原 watcher 若在 cotrain 结束时自动恢复 GPU1 R2R encoder，
会与 cmdcheck 抢显存。因此已停止 `nav_stage2_gate300_watch_20260817`，
改为手动收尾：

1. 等 cotrain 300-step 完成；
2. 等 legacy20 cmdcheck 20-step 完成；
3. 运行 `scripts/analyze_stage2_gate.py` 生成 `report_final.json`；
4. `SIGCONT` 恢复 `3936707,3936711` 两个 R2R obs-latent encoder。

2026-08-17 06:12 final report 已生成：

```text
report:
  NAV/result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json

paired train visual:
  common logged steps = 61
  mean(cotrain_visual - videoonly_visual) = +0.00041
  max_abs_diff = 0.01373

last5 train visual mean:
  cotrain/video-only = 0.23593 / 0.23723

last10 train visual mean:
  cotrain/video-only = 0.25065 / 0.25266
```

eval/loss_visual：

| step | cotrain | video-only |
| ---: | ---: | ---: |
| 50 | 0.43922 | 0.43414 |
| 100 | 0.35400 | 0.35079 |
| 150 | 0.31832 | 0.31861 |
| 200 | 0.24091 | 0.23924 |
| 250 | 0.23319 | 0.23505 |
| 300 | 0.41155 | 0.41321 |

cotrain eval/loss_pose：

```text
step50  = 0.03918
step100 = 0.01973
step150 = 0.00329
step200 = 0.00232
step250 = 0.01880
step300 = 0.00371
```

结论：300-step gate 通过。加入 Stage2 pose supervision 后，video generation
loss 与同结构 video-only 在 train 和 eval 上基本重合；pose loss 虽有单点评估
回弹，但整体从 0.039 量级降到 0.003–0.004 量级。因此当前 two-pass
legacy20-compatible graft 满足“3D 监督不破坏生成支路收敛”的短程/中程验证。

### Stage2 RE10K-full 2000-step cotrain run（2026-08-17）

300-step gate 通过后，启动更长的 Stage2 co-convergence 验证。该 run 仍只使用
当前落盘可用的 RE10K latent/pose 数据，因此是 **RE10K-only 结构验证**，
不是最终多数据规模训练。

```text
run:
  log/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/

tmux:
  nav_stage2_re10kfull_2k_iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532

TensorBoard:
  nav_stage2_re10kfull_2k_tb_6024
  port = 6024
  logdir = run/tensorboard only

GPU:
  cuda:1
  R2R obs-latent encoder shard-001 is SIGSTOP while this run trains
  shard-000 on cuda:0 continues preparing data

resume watcher:
  nav_stage2_re10kfull_2k_resume_gpu1_encoder
  behavior: wait for train PID 3954754 to exit, then SIGCONT encoder PID 3936711
```

训练参数：

```text
steps = 2000
batch_size = 1
grad_accum = 1
effective_batch_size = 1
history_steps = 1
lambda_pose = 0.1
save_every = 500
eval_every = 100
max_train_samples = 0  # use all currently available RE10K episodes
dataset_size = 269
Wan loaded keys = 825
```

启动 sanity：

```text
step1:
  train/loss_visual = 1.4064984321594238
  train/loss_pose   = 0.423828125
  cuda_max_memory   ≈ 27.97 GiB allocated / 35.6 GiB nvidia-smi used

step10:
  train/loss_visual = 0.8976835608482361
  train/loss_pose   = 0.34765625

step20:
  train/loss_visual = 0.5455908179283142
  train/loss_pose   = 0.2255859375

step40:
  train/loss_visual = 0.3487238585948944
  train/loss_pose   = 0.044189453125

step100:
  train/loss_visual = 0.32501161098480225
  train/loss_pose   = 0.0064697265625

eval@100:
  eval/loss_visual = 0.31304841488599777
  eval/loss_pose   = 0.0068531036376953125

step200:
  train/loss_visual = 0.232156902551651
  train/loss_pose   = 0.0026397705078125

eval@200:
  eval/loss_visual = 0.2013373076915741
  eval/loss_pose   = 0.01790904998779297
```

与 300-step gate 中 cotrain `eval@100 visual=0.35400` 相比，本次 RE10K-full
2k run 的 `eval@100 visual=0.31305` 更低；pose eval 也从 `0.01973`
降到 `0.00685`。因此截至 step100，该长程 cotrain 继续支持“3D 监督不破坏
生成支路，并且 pose probe 能同步收敛”的判断。

截至 step200，`eval/loss_visual` 继续下降到 `0.20134`；`eval/loss_pose`
从 step100 的 `0.00685` 回弹到 `0.01791`，但 train pose 仍为低量级
`0.00264`。当前判断：video generation branch 长程收敛趋势健康，pose
probe 存在采样波动但没有发散。下一关键检查点是 step500 checkpoint 和专用
eval 脚本的 decoded proxy video/pose 细项评估。

2026-08-17 07:36 状态：训练仍在运行，已到 step310；step500 checkpoint 尚未
产生。专用 eval 脚本已就绪，并已挂自动 watcher：

```text
tmux:
  nav_stage2_re10kfull_2k_step500_evalwatch

behavior:
  wait for checkpoints/step_000500.pt
  SIGSTOP GPU0 R2R obs-latent encoder PID 3936707
  run IWAligned Stage2 eval on cuda:0
  SIGCONT encoder after eval exits

watcher log:
  log/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/
      eval_step500_watcher.log
```

截至 step300 的训练/评估趋势：

```text
50-step train window means:
  step001-050 visual_mean=0.683492 pose_mean=0.193024
  step051-100 visual_mean=0.362902 pose_mean=0.005278
  step101-150 visual_mean=0.315060 pose_mean=0.003314
  step151-200 visual_mean=0.219391 pose_mean=0.019508
  step201-250 visual_mean=0.211341 pose_mean=0.001985
  step251-300 visual_mean=0.186312 pose_mean=0.001350

eval:
  step100 visual=0.313048 pose=0.006853
  step200 visual=0.201337 pose=0.017909
  step300 visual=0.171676 pose=0.002598
  step400 visual=0.140438 pose=0.002431

latest:
  step310 train/loss_visual=0.102795
  step310 train/loss_pose=0.000496
  step330 train/loss_visual=0.171616
  step330 train/loss_pose=0.001495
  step340 train/loss_visual=0.146731
  step340 train/loss_pose=0.002197
  step350 train/loss_visual=0.163023
  step350 train/loss_pose=0.001389
  step360 train/loss_visual=0.169995
  step360 train/loss_pose=0.006653
  step370 train/loss_visual=0.158398
  step370 train/loss_pose=0.003387
  step380 train/loss_visual=0.207254
  step380 train/loss_pose=0.001190
  step390 train/loss_visual=0.090179
  step390 train/loss_pose=0.001640
  step400 train/loss_visual=0.152338
  step400 train/loss_pose=0.017090
  step410 train/loss_visual=0.143577
  step410 train/loss_pose=0.002380
  step420 train/loss_visual=0.185941
  step420 train/loss_pose=0.001541
  step430 train/loss_visual=0.178283
  step430 train/loss_pose=0.001694
  step440 train/loss_visual=0.129776
  step440 train/loss_pose=0.000584
  step450 train/loss_visual=0.156292
  step450 train/loss_pose=0.001572
```

当前判断：longer Stage2 cotrain 不仅没有破坏 video branch，`loss_visual`
相较 step1/10/20/100 持续下降；pose probe 也从 0.42 量级降到
1e-3–1e-4 量级。下一硬证据仍是 step500 checkpoint 的 decoded proxy
video 与 pose 细项评估。

2026-08-17 07:44 状态：训练进程健康继续，step350 已写入；速度仍约
`11.36s/optimizer step`。按该速度，第一个 `step_000500.pt` 预计约 28–30
分钟后产生；`nav_stage2_re10kfull_2k_step500_evalwatch` 会自动触发评估。
本轮额外确认 `scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py`
可以 `py_compile` 和模块导入，并会解析到 `Infinite-World-legacy20`。

2026-08-17 07:45 补充：为保证 2k run 的完整证据链，除 step500 watcher
外，已挂 step1000/1500/2000 顺序 eval watcher：

```text
tmux:
  nav_stage2_re10kfull_2k_evalwatch_1000_2000

watcher log:
  log/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/
      eval_1000_2000_watcher.log

behavior:
  wait for step_001000.pt, step_001500.pt, step_002000.pt sequentially
  for each checkpoint: SIGSTOP GPU0 R2R encoder, run same IWAligned eval,
  then SIGCONT encoder
```

2026-08-17 07:53 状态：`eval@400` 已写入。虽然 step380/400 的 train visual
存在单点波动，eval visual 从 step300 的 `0.17168` 继续降到 step400 的
`0.14044`；eval pose 维持在 `0.00243` 低量级。因此截至 step400，longer
Stage2 cotrain 仍支持“video generation branch 不退化，pose probe 同步收敛”
的判断。step500 checkpoint 尚未产生，等待自动 decoded-proxy eval。

预期验收：

1. video loss 的长期趋势不劣于 300-step gate 中的 cotrain 曲线；
2. pose eval 维持在低量级并总体下降；
3. step500/1000/1500/2000 checkpoint 可用于后续生成质量和 pose probe 评估；
4. 训练结束后恢复 GPU1 的 R2R obs-latent encoder。

checkpoint 评估入口：

```bash
python scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py \
  --checkpoint log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/checkpoints/step_000500.pt \
  --run-name iw_aligned_re10kfull_2k_step500_eval \
  --device cuda:0 \
  --batch-size 1 \
  --max-batches 32 \
  --decode-rgb \
  --decode-batches 4 \
  --save-videos 4
```

该脚本为 `IWAlignedWorldNavModel` 专用，读取同一个 two-pass
`forward_stage2()`，指标包括 velocity MSE、one-step RFlow x0 proxy latent
metrics、pose translation/rotation/FOV，以及可选 VAE decode 的 RGB proxy 视频。

2026-08-17 08:18 更新：`step_000500.pt` 已产生，训练继续运行。训练内日志：

```text
step480 train/loss_visual = 0.7376417517662048  # 单点 spike
step490 train/loss_visual = 0.1902635544538498
step500 train/loss_visual = 0.10883779078722
step500 train/loss_pose   = 0.000949859619140625

eval@500:
  eval/loss_visual = 0.2380673922598362
  eval/loss_pose   = 0.003208160400390625
```

`nav_stage2_re10kfull_2k_step500_evalwatch` 首次触发时暴露了两个 evaluator
兼容问题，均已修复到
`scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py`：

1. checkpoint 中没有未使用的 `backbone.latent_encoder.*` 权重；eval 现在先
   `model.remove_hmpc()`，与训练保存态对齐；
2. `load_state_dict(strict=False)` 只允许 `backbone.latent_encoder.*` missing，
   其他 missing/unexpected 仍报错，避免吞掉真实结构错配。

修复后已手动补跑 step500 decoded eval，输出为：

```text
result/v1_stage2_iw_aligned_re10k_pose_video/
  iw_aligned_re10kfull_2k_step500_eval/
```

完整指标：

```text
eval/loss_total                  = 0.15479633072391152
eval/loss_visual_velocity_mse    = 0.15447463025338948
eval/loss_pose_mse               = 0.003216995095499442
eval/latent_x0_mse               = 0.07880453507459606
eval/latent_x0_rmse              = 0.27066083753015846
eval/latent_x0_l1                = 0.20447151694679633
eval/baseline_noisy_latent_mse   = 1.0786058899502677
eval/latent_mse_improvement_vs_noisy = 0.999801362209837
eval/latent_x0_cosine            = 0.931394575163722
eval/pose_translation_mae        = 0.034744832722935826
eval/pose_rotation_angle_deg     = 7.362439222633839
eval/pose_fov_mae                = 0.06077679945155978
eval_decode/rgb_mse              = 0.008316722116433084
eval_decode/rgb_l1               = 0.053823224268853664
eval_decode/rgb_psnr_db          = 21.63981533050537
eval_decode/rgb_pred_mean/std    = 0.519428513944149 / 0.24406779929995537
eval_decode/rgb_target_mean/std  = 0.518842525780201 / 0.25330356508493423
eval/num_batches                 = 32
eval_decode/num_batches          = 4
```

视频可播放性/非黑屏检查：

```text
samples: 4 groups, each has obs.mp4 / gt_future.mp4 / pred_future_proxy.mp4
format : 896x448, 13 frames, 12 fps, duration 1.083s
pred mean range   ≈ 0.4266 - 0.5822
pred std range    ≈ 0.2175 - 0.2972
pred motion_l1    ≈ 0.0163 - 0.0272
```

结论：截至 step500，Stage2 RE10K-only longer cotrain 已经给出三层证据：
训练 loss 持续下降、pose probe 低量级、decoded proxy RGB 非黑屏且有像素运动。
`eval@500` 的训练内 visual loss 相比 `eval@400` 有回弹，但手动 32-batch
decoded eval 的 velocity MSE 为 `0.15447`，仍处于健康区间。后续继续观察
step1000/1500/2000 自动 eval。

2026-08-17 08:22 加固：为避免后续 checkpoint 再出现 HPMC runtime state
歧义，`scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py` 与
`scripts/train_v1_stage3_iw_aligned_r2r_policy.py` 的新 checkpoint 会额外保存：

```text
runtime_flags:
  hmpc_removed
  backbone_use_convenc
```

`scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py` 读取旧 checkpoint 时默认
按既有 IW-aligned 训练态 `remove_hmpc()`；读取未来带 `runtime_flags` 的
checkpoint 时按 flag 恢复。已通过 `py_compile` 和旧 `step_000500.pt` CPU
加载回归：

```text
loaded_step = 500
hmpc_removed = True
has_runtime_flags = False  # 旧 checkpoint，按默认兼容
```

08:22 运行态：

```text
step550 train/loss_visual = 0.151582270860672
step550 train/loss_pose   = 0.002899169921875
train PID 3954754 healthy on cuda:1
GPU0 R2R encoder PID 3936707 running
GPU1 R2R encoder PID 3936711 still SIGSTOP until Stage2 train exits
```

08:24 运行态：

```text
step560 train/loss_visual = 0.09157153218984604
step560 train/loss_pose   = 0.0024566650390625
step_001000.pt not yet produced
eval_1000_2000_watcher remains alive and waiting for step_001000.pt
```

当前 step500/560 证据只证明 IWAligned legacy20-compatible graft 上的
generation+pose cotrain 健康；不证明 Stage3 full-cache policy，也不证明
formal 16ch/no-mask V1。模型层面的证据边界已同步到
`doc/01_model/model_evolution_and_current_architecture.md`。

08:26 运行态：

```text
step570 train/loss_visual = 0.0701913833618164
step570 train/loss_pose   = 0.0024566650390625
recent speed              ≈ 11.33s / optimizer step
ETA to step1000           ≈ 81 min + decoded eval time
```

按 100-step 窗口统计：

```text
step001-100 visual_mean=0.537769 pose_mean=0.107685
step101-200 visual_mean=0.267226 pose_mean=0.011411
step201-300 visual_mean=0.198827 pose_mean=0.001668
step301-400 visual_mean=0.149791 pose_mean=0.003667
step401-500 visual_mean=0.208113 pose_mean=0.002057  # 被 step480 spike 拉高
step501-600 visual_mean=0.154912 pose_mean=0.002267  # 当前只到 step570
```

判断：`eval@500` 有 visual 回弹，但 step501-570 训练窗口没有发散，且
`step570` visual 已降到 `0.07019`。继续等待 step1000 decoded eval。

08:27 运行态：

```text
step580 train/loss_visual = 0.06224000081419945
step580 train/loss_pose   = 0.00156402587890625
step_001000.pt still not produced
eval_1000_2000_watcher still waiting for step_001000.pt
```

判断：step580 继续下降，当前没有训练崩溃或 pose/generation 明显冲突迹象。

这条 300-step gate 完成后应使用 `analyze_stage2_gate.py` 生成正式报告，并重点看：

- paired train visual mean/max difference；
- eval visual at step50/100/150/200/250/300；
- cotrain pose eval 是否继续下降；
- 是否开始接近 legacy20 旧曲线的 0.5 左右平台。

注意：这仍是链路验证，不是 policy 性能证明。下一步必须把
`obs_latents_r2r_smoke` 替换成 full R2R obs latent cache，并启动更长的 Stage3
policy 训练；同时保持 Stage2/Stage1 replay gate，监控 generation loss 不退化。

同时为回应“旧版命令再训练一下看收敛速度”，已再次用旧版命令做
legacy20 Stage1 cmdcheck。注意：直接调用训练脚本时必须显式设置
`NAV_INF_WORLD_ROOT=Infinite-World-legacy20`；否则会误指向当前新版
`Infinite-World`，触发 `16/20 channel mismatch`。包装脚本
`scripts/run_v1_wan_stage1_t4_history.sh` 已默认锁定 legacy20 root。

```text
rerun:
  log/legacy20_stage1_t4_fullmix_cmdcheck_rerun_20260817_072137/

command semantics:
  scripts/train_v1_wan_stage1_t4_history.py
  --device cuda:0
  --history-iw-chunks 1,4,8,16
  --data-root /sharedata/NAV/derived/v1/t4_micro_latents_spatial20
  --checkpoint /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
  --variant latent_prefix
  --train-scope full
  --batch-size 1
  --gradient-accumulation-steps 16
  --steps 10
  --log-every 1
```

observed rerun:
  step1  = 1.6130983456969261
  step5  = 0.8757058940827847
  step10 = 0.8148440010845661

legacy20 baseline 20-step:
  log/legacy20_stage1_t4_fullmix_convergence_20260817_021207/
  step1  = 1.6130983456969261
  step5  = 0.8752648654580116
  step10 = 0.8151123039424419
  step20 = 0.5724489018321037
```

结论：旧版有效 generation branch 的命令、数据采样、seed、legacy20 backbone、
20-channel patch stem、Register/history 构造和 text cache 均再次对齐。rerun
在 step10 已与旧基线基本重合：`0.814844` vs `0.815112`，step1→step10
约下降 `49.5%`。因此该收敛速度确认成立，不需要为了同一结论继续抢占当前
Stage2/R2R 数据准备用卡。

2026-08-17 08:32 又按同一旧命令启动一次完整路径短跑：

```text
log/legacy20_stage1_t4_fullmix_cmdtrain_20260817_083255/
```

本次显式使用：

- `NAV_INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20`
- `cuda:0`
- `history_iw_chunks=1,4,8,16`
- `batch_size=1`
- `gradient_accumulation_steps=16`
- `steps=30`

结果：

```text
step1 loss = 1.6130983456969261
history_iw_chunks = 16
history_micro_steps = 107
peak_reserved_gib = 31.2890625
```

该 step1 与 `021207`、`024822`、`072137` 三条旧命令记录完全一致，进一步确认
入口、seed、数据采样和 legacy20 结构没有跑偏。但这次在 step1 后等待约 11
分钟仍未写出 step2，GPU0 持续 100% util、显存约 38.7GiB。为避免影响 R2R
obs latent 准备，已停止该 cmdtrain 并恢复 GPU0 编码进程。

因此后续比较“旧版收敛速度”时使用已完成的 `021207/024822/072137` 数值曲线：

- step1→step5：`1.6131 → 0.8753`，下降约 `45.7%`；
- step1→step10：`1.6131 → 0.8150`，下降约 `49.5%`；
- step1→step20：`1.6131 → 0.5724`，下降约 `64.5%`。

注意：这里的“收敛快”指 optimize-step loss 斜率快；旧命令每个 optimize step
包含 `gradient_accumulation_steps=16` 个完整样本，且可能抽到 IW=16 的
107-micro history，因此 wall-clock 并不轻。

2026-08-17 09:08 再按同一旧命令做了 10-step 收敛速度复跑，用于确认
“上一版本命令”在当前环境中仍可快速下降：

```text
log/legacy20_stage1_prevcmd_20step_20260817_090828/

NAV_INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20
device=cuda:0
history_iw_chunks=1,4,8,16
batch_size=1
gradient_accumulation_steps=16
effective_batch_size=16
target_latent_t=4
```

该 run 原计划 20 step，但拿到 step10 后已手动停止并恢复 GPU0 R2R
obs latent encoder。前 10 个 optimizer step 与旧 baseline 再次逐点对齐：

| step | loss | history IW chunks | history micro steps | peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.613098 | 16 | 107 | 31.29 |
| 2 | 1.692860 | 16 | 107 | 37.31 |
| 3 | 1.222753 | 1 | 7 | 37.31 |
| 4 | 1.141407 | 4 | 27 | 37.31 |
| 5 | 0.875129 | 16 | 107 | 37.31 |
| 6 | 0.983618 | 8 | 54 | 37.31 |
| 7 | 0.891424 | 8 | 54 | 37.31 |
| 8 | 0.946757 | 1 | 7 | 37.31 |
| 9 | 0.849615 | 16 | 107 | 37.31 |
| 10 | 0.814849 | 16 | 107 | 37.31 |

统计：step1→step10 从 `1.613098` 降到 `0.814849`，下降约 `49.5%`；
按 run name 到 step10 log mtime 粗略估计 wall-clock 为 `741.8s`，
平均约 `74.2s / optimizer step`。由于每个 optimizer step 有 16 次
micro accumulation，且 history 长度随机，wall-clock 不能直接和固定短历史/
纯 Stage2 单样本训练比较。

2026-08-17 08:49 继续推进 final-setting graft：

代码改动：

```text
src/nav/v1/models/iw_aligned.py
scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py
scripts/train_v1_stage3_iw_aligned_r2r_policy.py
scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py
```

新增真实 IWAligned Register mode：

```text
--register-mode legacy_extract_update   # 默认，旧版快速收敛路径
--register-mode fixed_rnull_unified     # final V1 graft：fixed R_null + unified recurrent updater
```

新增真实 IWAligned A_cur routing mode：

```text
--current-action-mode legacy_iw_move_view  # 默认，旧版 IW native move/view video action embedding
--current-action-mode shared_tail_token    # final V1 graft：禁用 native action embedding，A_cur 作为 shared Wan tail condition token
--current-action-mode none                 # 诊断用，无 A_cur 条件
```

对应可复用启动脚本：

```bash
bash scripts/run_v1_stage2_register_mode_gate.sh fixed_rnull_unified cuda:0 300
bash scripts/run_v1_stage2_register_mode_gate.sh legacy_extract_update cuda:0 300
bash scripts/run_v1_stage2_register_mode_gate.sh fixed_rnull_unified cuda:0 300 formal_acur_gate shared_tail_token
```

为了避免 fixed-rnull gate 抢占 step1000 decoded eval 的 GPU0，已挂一个顺序
watcher：

```text
tmux session = nav_fixed_rnull_gate_after_step1000
script       = scripts/watch_step1000_then_run_fixed_rnull_gate.sh
log          = log/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_gate_after_step1000_watcher.log
wait target  = result/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_re10kfull_2k_step1000_eval/eval_metrics.json
then run     = bash scripts/run_v1_stage2_register_mode_gate.sh fixed_rnull_unified cuda:0 300 <auto_run_name>
then analyze = scripts/analyze_register_mode_gate.py
report path  = result/v1_stage2_iw_aligned_re10k_pose_video/register_mode_gate_<auto_run_name>/report.json
```

该 watcher 只有在 step1000 decoded metrics 已落盘后才暂停 GPU0 R2R encoder
`3936707` 并启动 fixed-rnull gate；结束后会恢复 encoder。预计不会撞上
step1500 decoded eval，因为 step1000→step1500 还有约 500 个训练 step 的窗口。

`scripts/analyze_register_mode_gate.py` 会把 fixed-rnull candidate 与当前
`iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532` 的前 300 step legacy
baseline 逐 step 对齐，输出：

- `mean_visual_diff_candidate_minus_baseline`
- `max_abs_visual_diff`
- `mean_pose_diff_candidate_minus_baseline`
- last10 paired loss table

脚本已做 self-compare 自检：baseline 与自身比较时 `mean_visual_diff=0`、
`max_abs_visual_diff=0`。

`fixed_rnull_unified` 的语义：

- `R0` 是非学习的 zero tensor，形状仍为 `[B,16,4,H,W]`，可直接作为
  legacy20 Wan 的 `image_cond` prefix；
- 第一个 history chunk 和所有后续 history chunk 都通过同一组 `unified_blocks`
  更新 Register；
- `A_hist` 仍作为 latent-channel context tokens 进入 Register cross-attention；
- 不改变 video pass：仍是 `image_cond=Register`、`local_memory=z_obs latest`、
  Wan native future velocity head。

兼容性检查：

```text
py_compile:
  src/nav/v1/models/iw_aligned.py
  scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py
  scripts/train_v1_stage3_iw_aligned_r2r_policy.py
  scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py

structure instantiation:
  legacy_extract_update ok
  fixed_rnull_unified ok

old checkpoint compatibility:
  step_000500.pt CPU load ok
  register_mode = legacy_extract_update
  hmpc_removed = true
```

由于当前 2k Stage2 run 是旧进程启动的，后续它保存的 step1000/1500/2000
checkpoint 仍不会包含新增的 `register_memory.unified_blocks.*` 参数。evaluator
已允许旧 checkpoint 仅在 `legacy_extract_update` 语义下缺失这些 opt-in 参数，避免
自动 decoded eval watcher 再次因 checkpoint schema 演进失败；如果未来
`fixed_rnull_unified` checkpoint 缺失 `unified_blocks`，仍会报错，不能静默用随机
初始化参数评测。

额外 preflight：

```text
run = log/v1_stage2_iw_aligned_re10k_pose_video/stage2_register_gate_fixed_rnull_unified_preflight_20260817_085300/
status = ok
dataset_size = 4
register_mode = fixed_rnull_unified
history_latents = [1,16,4,56,112]
z_obs = [16,4,56,112]
z_future_noisy/target = [16,4,56,112]
pose_target = [4,9]
```

该 preflight 不跑模型、不抢 GPU，只证明真实 RE10K T4 数据与
`fixed_rnull_unified` gate 入口可构建。

formal A_cur shared-tail preflight：

```text
run = log/v1_stage2_iw_aligned_re10k_pose_video/stage2_current_action_shared_tail_preflight_20260817_090218/
status = ok
dataset_size = 4
register_mode = fixed_rnull_unified
current_action_mode = shared_tail_token
history_latents = [1,16,4,56,112]
z_future_noisy/target = [16,4,56,112]
pose_target = [4,9]
```

额外结构检查：

```text
current_action_mode construct_ok:
  legacy_iw_move_view
  shared_tail_token
  none
```

注意：`shared_tail_token` 只能在 `fixed_rnull_unified` 单独通过 generation gate
后再单独开 gate。当前 step1000 后自动触发的仍是 fixed-rnull-only gate：
`current_action_mode=legacy_iw_move_view`，用于隔离 Register 改动影响。

当前运行态：

```text
Stage2 2k cotrain:
  run = log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/
  latest train step = 1020
  latest eval step = 1000
  step680 visual = 0.08076603710651398, pose = 0.0021209716796875
  step690 visual = 0.12269198894500732, pose = 0.004486083984375
  step700 train/loss_visual = 0.11591298133134842
  step700 train/loss_pose   = 0.0008392333984375
  step700 eval/loss_visual  = 0.15353064611554146
  step700 eval/loss_pose    = 0.0071048736572265625
  step720 train/loss_visual = 0.05699655041098595
  step720 train/loss_pose   = 0.0013275146484375
  step750 train/loss_visual = 0.4785309135913849
  step760 train/loss_visual = 0.11963463574647903
  step770 train/loss_visual = 0.14369729161262512
  step770 train/loss_pose   = 0.0018157958984375
  step780 train/loss_visual = 0.4082074761390686
  step780 train/loss_pose   = 0.0076904296875
  step800 train/loss_visual = 0.0789838582277298
  step800 train/loss_pose   = 0.0026092529296875
  step800 eval/loss_visual  = 0.1578779574483633
  step800 eval/loss_pose    = 0.0041637420654296875
  step830 train/loss_visual = 0.15020067989826202
  step830 train/loss_pose   = 0.0169677734375
  step870 train/loss_visual = 0.08950906991958618
  step870 train/loss_pose   = 0.00122833251953125
  step900 train/loss_visual = 0.6804605722427368
  step900 train/loss_pose   = 0.0012054443359375
  step900 eval/loss_visual  = 0.07458615023642778
  step900 eval/loss_pose    = 0.018332481384277344
  step940 train/loss_visual = 0.06412886828184128
  step940 train/loss_pose   = 0.00099945068359375
  step990 train/loss_visual = 0.0543195866048336
  step990 train/loss_pose   = 0.007110595703125
  step1000 train/loss_visual = 0.047473400831222534
  step1000 train/loss_pose   = 0.00421142578125
  step1000 eval/loss_visual  = 0.061783477663993835
  step1000 eval/loss_pose    = 0.0021066665649414062
  recent logged steps 901-1000:
    train/loss_visual mean = 0.09638154245913029
    train/loss_pose mean   = 0.0029483795166015624
  seconds_per_step_window ≈ 11.37s
  step1000 checkpoint = checkpoints/step_001000.pt
  step1500 checkpoint pending

R2R obs latent:
  shard0 = 2870 episodes
  shard1 = 773 episodes
  total = 3643 / 13436 episodes
  disk = 39G
  current state = shard0/shard1 are SIGSTOP while GPU0 fixed-rnull gate and GPU1 Stage2 run
```

判断：`eval@600` 的 visual 曾回弹到 `0.3662`，但 `eval@700` 已恢复到
`0.1535`，`eval@800` 继续稳定在 `0.1579`，`eval@900` 降到 `0.0746`，
`eval@1000` 进一步到 `0.0618`。说明 step500/600 之后的波动目前更像
采样/训练噪声，而不是 generation 分支被 pose loss 破坏。step750/780/830/890/900
均出现过 train visual 或 pose spike，但后续又回落；到 step1000，video generation
和 pose probe 均处于目前最好的验证区间。

Step1000 decoded eval：

```text
result/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_re10kfull_2k_step1000_eval/

checkpoint = log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/checkpoints/step_001000.pt
max_batches = 32
decode_rgb = true
decode_batches = 4
save_videos = 4
```

关键指标：

| metric | value |
| --- | ---: |
| `eval/loss_visual_velocity_mse` | 0.10243806533981115 |
| `eval/loss_pose_mse` | 0.0029859624883101787 |
| `eval/latent_x0_mse` | 0.06472588352335151 |
| `eval/latent_x0_cosine` | 0.94505824893713 |
| `eval/pose_translation_mae` | 0.031238393508829176 |
| `eval/pose_rotation_angle_deg` | 6.110842011868954 |
| `eval/pose_fov_mae` | 0.0657082861289382 |
| `eval_decode/rgb_psnr_db` | 25.348965167999268 |
| `eval_decode/rgb_pred_mean` | 0.5248059630393982 |
| `eval_decode/rgb_pred_std` | 0.24821478500962257 |
| `eval_decode/rgb_target_mean` | 0.518842525780201 |
| `eval_decode/rgb_target_std` | 0.25330356508493423 |

视频有效性检查：4 组 sample，每组 `obs.mp4 / gt_future.mp4 / pred_future_proxy.mp4`
均为 `896x448`、`13 frames`、`12 fps`，全部可由 `ffprobe/OpenCV` 读取；
所有 `pred_future_proxy.mp4` 的 mean/std/temporal_absdiff 均非黑屏、非静态。

fixed-rnull gate：

```text
log/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743/

register_mode = fixed_rnull_unified
current_action_mode = legacy_iw_move_view
device = cuda:0
steps = 300
```

该 gate 已在 step1000 decoded metrics 落盘后自动启动。首个 step：

```text
step1 train/loss_visual = 1.4576925039291382
step1 train/loss_pose   = 0.59765625
peak memory ≈ 27.97GiB
```

解释：fixed-rnull gate 是从同一 Wan2.1 官方权重初始化、但 Register
初始化/更新规则不同的新 run，不是从 step1000 checkpoint 续训；其用途是隔离
“先固定 Rnull，再统一 updater”的 Register 改动是否破坏旧 generation branch。

fixed-rnull gate 早期进展：

| step | legacy baseline train visual | fixed-rnull train visual | diff | legacy baseline train pose | fixed-rnull train pose |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4064984321594238 | 1.4576925039291382 | +0.051194071769714355 | 0.423828125 | 0.59765625 |
| 10 | 0.8976835608482361 | 0.865523099899292 | -0.03216046094894409 | 0.34765625 | 0.37109375 |
| 20 | 0.5455908179283142 | 0.7155718207359314 | +0.1699810028076172 | 0.2255859375 | 0.1806640625 |
| 30 | 0.5225532650947571 | 0.9366363286972046 | +0.4140830636024475 | 0.1015625 | 0.087890625 |
| 40 | 0.3487238585948944 | 0.49600791931152344 | +0.14728406071662903 | 0.044189453125 | 0.017578125 |
| 50 | 0.379901260137558 | 0.42271265387535095 | +0.04281139373779297 | 0.01531982421875 | 0.0167236328125 |

fixed-rnull `eval@50`：

```text
eval/loss_visual = 0.4761275574564934
eval/loss_pose   = 0.035430908203125
```

临时判断：fixed-rnull 的 train visual 在 step50 已基本追近 legacy baseline，
pose 也下降正常；但 step30 有明显回弹，`eval@50` 仍偏高。因此该 gate 不能提前
判定通过，需要至少等 step100 eval 和 300-step analyzer report。

fixed-rnull `step100` 更新：

| step | legacy baseline train visual | fixed-rnull train visual | diff | legacy baseline train pose | fixed-rnull train pose |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 0.4549286663532257 | 0.2859324514865875 | -0.16899621486663818 | 0.007537841796875 | 0.0059814453125 |
| 70 | 0.39530497789382935 | 0.5591363310813904 | +0.16383135318756104 | 0.00250244140625 | 0.004241943359375 |
| 80 | 0.24802173674106598 | 0.29400551319122314 | +0.045983776450157166 | 0.00811767578125 | 0.0045166015625 |
| 90 | 0.39124172925949097 | 0.3498344421386719 | -0.04140728712081909 | 0.00176239013671875 | 0.005584716796875 |
| 100 | 0.32501161098480225 | 0.31476470828056335 | -0.010246902704238892 | 0.0064697265625 | 0.00616455078125 |

分段均值：

| steps | legacy baseline visual mean | fixed-rnull visual mean | diff |
| --- | ---: | ---: | ---: |
| 1–50 | 0.6834918657938639 | 0.81569072107474 | +0.13219885528087616 |
| 51–100 | 0.36290174424648286 | 0.36073468923568724 | -0.0021670550107956155 |

eval 对比：

| step | legacy baseline eval visual | fixed-rnull eval visual | legacy baseline eval pose | fixed-rnull eval pose |
| ---: | ---: | ---: | ---: | ---: |
| 50 | - | 0.4761275574564934 | - | 0.035430908203125 |
| 100 | 0.31304841488599777 | 0.28971344232559204 | 0.0068531036376953125 | 0.02368927001953125 |

更新判断：fixed-rnull 前 50 step 更慢且更抖，但 51–100 step 已追平 legacy
baseline，`eval@100` visual 略低于 baseline；pose eval 偏高但仍在下降量级。
因此不应提前杀该 gate，应继续等 300-step analyzer report 再决定是否接受
`fixed_rnull_unified` 为正式 Register 初始化/更新规则。

fixed-rnull `step150` 更新：

| steps | legacy baseline train visual mean | fixed-rnull train visual mean | diff | legacy baseline train pose mean | fixed-rnull train pose mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101–150 | 0.315060 | 0.297659 | -0.017401 | 0.003314 | 0.003073 |

| step | fixed-rnull train visual | fixed-rnull train pose | fixed-rnull eval visual | fixed-rnull eval pose |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 0.2850947678089142 | 0.0025787353515625 | 0.23710491508245468 | 0.002620697021484375 |

截至 step150，fixed-rnull 已从前 50 step 的慢启动恢复到与 legacy baseline
同量级，且 101–150 的 train visual 均值略低于 baseline；pose train/eval 也
降到 `~0.0026–0.0031`。当前证据支持继续跑满 300-step gate，但仍不能替代
300-step analyzer report 与 decoded RGB 检查。

为了补齐 decoded RGB 验收，已新增并启动后台 watcher：

```text
tmux session = nav_fixed_rnull_decode_after300
script       = scripts/watch_fixed_rnull_step300_eval_decode.sh
wait ckpt    = log/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743/checkpoints/step_000300.pt
eval output  = result/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743_step300_decode_eval/
log          = log/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743/eval_step300_decode_watcher.log
```

该 watcher 只等待 checkpoint，不修改训练；checkpoint 出现且训练进程退出后，
它会临时暂停 GPU0 的 R2R latent worker，按 step1000 主 run 相同 evaluator
跑 `max_batches=32`、`decode_batches=8`、`save_videos=4`，随后恢复 worker。

另外新增通用视频检查脚本：

```text
script       = scripts/validate_eval_videos.py
tmux session = nav_fixed_rnull_video_validate_after300
output       = result/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743_step300_decode_eval/video_validity.json
log          = log/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_unified_after_step1000_gate_20260817_085743/video_validate_watcher.log
```

该检查会遍历 decoded eval 目录下所有 `mp4`，记录 frame count、mean/std、
temporal absdiff，并把不可读、近黑屏或近静态视频标记为 bad。

fixed-rnull `step180` 更新：

| step | fixed-rnull train visual | fixed-rnull train pose |
| ---: | ---: | ---: |
| 160 | 0.1436436027288437 | 0.01904296875 |
| 170 | 0.306953489780426 | 0.00531005859375 |
| 180 | 0.17247429490089417 | 0.00213623046875 |
| 190 | 0.19276879727840424 | 0.0031890869140625 |
| 200 | 0.22340644896030426 | 0.002685546875 |

fixed-rnull `eval@200`：

```text
eval/loss_visual = 0.21964940801262856
eval/loss_pose   = 0.0019159317016601562
```

分段均值扩展到 step200：

| steps | legacy baseline visual mean | fixed-rnull visual mean | diff | legacy baseline pose mean | fixed-rnull pose mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–50 | 0.683492 | 0.815691 | +0.132199 | 0.193024 | 0.211934 |
| 51–100 | 0.362902 | 0.360735 | -0.002167 | 0.005278 | 0.005298 |
| 101–150 | 0.315060 | 0.297659 | -0.017401 | 0.003314 | 0.003073 |
| 151–200 | 0.219391 | 0.207849 | -0.011541 | 0.019508 | 0.006473 |

eval 对比更新：

| step | legacy baseline eval visual | fixed-rnull eval visual | legacy baseline eval pose | fixed-rnull eval pose |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.31304841488599777 | 0.28971344232559204 | 0.0068531036376953125 | 0.02368927001953125 |
| 200 | 0.2013373076915741 | 0.21964940801262856 | 0.01790904998779297 | 0.0019159317016601562 |

当前 fixed-rnull 仍在训练中，GPU0 占用约 `35.9GiB`，每 10 step 窗口约
`10.9–12.0s/step`。R2R obs latent worker `3936707` 仍保持 SIGSTOP，等待
fixed-rnull gate / decoded eval 完成后由 watcher 恢复。

判断：fixed-rnull 的训练曲线在 51–200 step 不比 legacy baseline 慢；
`eval@200` visual 略高于 baseline，但 pose 明显更低。现阶段仍是“继续观察”
而不是正式接受；最终看 step300 analyzer report 与 decoded RGB/video validity。

fixed-rnull `step250` 更新：

| step | fixed-rnull train visual | fixed-rnull train pose | fixed-rnull eval visual | fixed-rnull eval pose |
| ---: | ---: | ---: | ---: | ---: |
| 220 | 0.13531458377838135 | 0.0045166015625 | - | - |
| 230 | 0.49489670991897583 | 0.00323486328125 | - | - |
| 240 | 0.14872924983501434 | 0.0018463134765625 | - | - |
| 250 | 0.28596025705337524 | 0.0024261474609375 | 0.21744230389595032 | 0.020221710205078125 |

分段均值继续扩展：

| steps | legacy baseline visual mean | fixed-rnull visual mean | diff | legacy baseline pose mean | fixed-rnull pose mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 201–250 | 0.211341 | 0.268502 | +0.057160 | 0.001985 | 0.003043 |

解释：201–250 的 fixed-rnull train visual 受 step230 难样本 spike 影响，高于
legacy baseline；但 step240 已回落，`eval@250` visual 仍约 `0.217`，与
`eval@200` 同量级并略低。pose eval 在 step250 出现 spike，需要等 step300
和 decoded eval 综合判断。当前仍不提前通过 gate。

fixed-rnull `step300` 最终 gate 结果：

| 项 | legacy_extract_update baseline | fixed_rnull_unified candidate | 说明 |
| --- | ---: | ---: | --- |
| paired train common steps | 31 | 31 | 对齐 step 1,10,...,300 |
| paired mean visual diff | - | `+0.03694068496265719` | candidate - baseline；略慢但未发散 |
| paired max abs visual diff | - | `0.4140830636024475` | 早期 step30 / 难样本 spike 主导 |
| paired mean pose diff | - | `+0.0026450618620841733` | 小幅高 |
| eval@300 visual | `0.17167602106928825` | `0.1948702670633793` | candidate 略差 |
| eval@300 pose | `0.002597808837890625` | `0.001556396484375` | candidate 更好 |
| decoded eval visual velocity MSE | - | `0.19762781611643732` | candidate step300 decoded |
| decoded latent x0 cosine | - | `0.894826477393508` | candidate step300 decoded |
| decoded RGB PSNR | - | `21.24316954612732` | candidate step300 decoded |
| decoded pose translation MAE | - | `0.039911698549985886` | candidate step300 decoded |
| decoded pose rotation angle | - | `7.296071507036686 deg` | candidate step300 decoded |
| video validity | - | `12/12 ok, 0 bad` | 所有 obs/gt/pred mp4 可读、非黑、非静态 |

路径：

```text
analyzer report =
  result/v1_stage2_iw_aligned_re10k_pose_video/
    register_mode_gate_fixed_rnull_unified_after_step1000_gate_20260817_085743/report.json

decoded eval =
  result/v1_stage2_iw_aligned_re10k_pose_video/
    fixed_rnull_unified_after_step1000_gate_20260817_085743_step300_decode_eval/eval_metrics.json

video validity =
  result/v1_stage2_iw_aligned_re10k_pose_video/
    fixed_rnull_unified_after_step1000_gate_20260817_085743_step300_decode_eval/video_validity.json
```

结论：`fixed_rnull_unified` 记为 **弱通过 / engineering pass**。
它没有达到 legacy_extract_update 的同等平滑收敛，尤其 201–300 区间有若干
visual spike；但其 video generation branch 没有崩坏，`eval@300` 仍在可接受
量级，pose 监督可收敛，decoded RGB 可播放且非黑屏/非静态。因此可以进入下一
单变量 gate：只切换 `A_cur` routing 为 `shared_tail_token`，但不能把
fixed-rnull 说成“优于 legacy”或最终结构充分验证。

主 Stage2 2k run 到 step1200：

| step | train/loss_visual | train/loss_pose | eval/loss_visual | eval/loss_pose |
| ---: | ---: | ---: | ---: | ---: |
| 1100 | 0.059310 | 0.001457 | 0.06465875543653965 | 0.0059795379638671875 |
| 1200 | 0.05314558744430542 | 0.00323486328125 | 0.1582256006076932 | 0.00118255615234375 |

分段 train 均值：

| steps | train visual mean | train visual min | train visual max | train pose mean |
| --- | ---: | ---: | ---: | ---: |
| 1001–1100 | 0.082538 | 0.047484 | 0.154683 | 0.003909 |
| 1101–1200 | 0.075357 | 0.053146 | 0.104136 | 0.002165 |

解释：train visual 仍在低位继续下降；`eval@1200` visual 较 `eval@1000/1100`
回弹，需要等 step1500 decoded eval 与更长趋势判断，不能单点判定退化。

下一条必须跑的结构 gate：

```text
legacy_extract_update Stage2/Stage1 visual baseline
vs
fixed_rnull_unified Stage2/Stage1 visual+pose run

验收：
  1. train/eval loss_visual 不能明显慢于 legacy mode；
  2. pose loss 仍需下降；
  3. decoded RGB 不能黑屏/不可播放；
  4. 通过后才继续 graft formal A_cur condition / Stage3 full-cache policy。
```

下一步 `shared_tail_token` gate 的静态审计：

```text
register_mode        = fixed_rnull_unified 或 legacy_extract_update（取决于 fixed-rnull gate 结论）
current_action_mode  = shared_tail_token
```

代码路径确认：

- `current_action_mode=shared_tail_token` 会关闭 InfiniteWorld native
  `action_encoder(move/view)`，并把 `A_cur` 编成一个 shared Wan tail condition
  token；
- Register、history、local memory、video RFlow loss、pose probe loss、RE10K
  dataset 均不随该 flag 改变；
- shared tail 序列形如 `[A_noise/query tokens, A_cur condition token]`；
- `IWActionInterface.decode()` 显式裁剪 `hidden[:, :action_horizon]`，因此追加的
  `A_cur condition token` 不会被当成 `A_out` 监督或解码；
- Stage2 当前没有 action loss，所以 shared-tail gate 只检验该 condition token
  是否破坏 video/pose 收敛。

因此 shared-tail gate 可以作为 fixed-rnull 后的单变量改动；但必须等
fixed-rnull 的 step300 analyzer、decoded eval 和 video validity 都完成后再启动，
避免把 Register 变更和 A_cur routing 变更混在同一个结论里。

fixed-rnull 已于 2026-08-17 10:52 完成 weak engineering pass。下一步
shared-tail gate 已挂后台 watcher，但不会立刻抢 GPU0：

```text
tmux session = nav_shared_tail_gate_after_step1500
script       = scripts/watch_step1500_then_run_shared_tail_gate.sh
wait metric  = result/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_re10kfull_2k_step1500_eval/eval_metrics.json
candidate    = fixed_rnull_unified + shared_tail_token
baseline     = fixed_rnull_unified + legacy_iw_move_view
steps        = 300
```

调度理由：主 Stage2 2k run 即将到 step1500，已有 decoded eval watcher
会用 GPU0；shared-tail gate 等 step1500 decoded eval 完成后再暂停 GPU0 R2R
encoder `3936707`、启动训练、跑 analyzer、decoded eval 与 video validity，
最后恢复 encoder。

2026-08-17 10:55 状态：

```text
main Stage2 run:
  latest train step = 1350
  train/loss_visual = 0.05407126992940903
  train/loss_pose   = 0.000652313232421875
  latest eval step  = 1300
  eval/loss_visual  = 0.07825325895100832
  eval/loss_pose    = 0.0014753341674804688

shared-tail watcher:
  session = nav_shared_tail_gate_after_step1500
  state   = waiting for iw_aligned_re10kfull_2k_step1500_eval/eval_metrics.json

GPU/data state:
  GPU0 = R2R obs latent worker 3936707 resumed/running
  GPU1 = main Stage2 training; R2R worker 3936711 stays SIGSTOP
```

因此当前不是 shared-tail gate 失败或漏启动，而是有意等待 step1500 decoded
eval 避免 GPU0 冲突。

2026-08-17 10:57 状态更新：

```text
main Stage2 latest train step = 1360
main train/loss_visual        = 0.07466327399015427
main train/loss_pose          = 0.0009002685546875
main latest eval step         = 1300
main eval/loss_visual         = 0.07825325895100832
shared-tail watcher           = still waiting for step1500 decoded eval
```

按当前 `~11.4s/step` 速度，step1360 到 step1500 约需 25–30 分钟；之后
`eval_1000_2000_watcher` 会先使用 GPU0 跑 step1500 decoded eval，随后
`nav_shared_tail_gate_after_step1500` 才启动 shared-tail gate。

2026-08-17 11:01 状态更新：

```text
main Stage2 latest train step = 1380
main train/loss_visual        = 0.07193649560213089
main train/loss_pose          = 0.000423431396484375
main latest eval step         = 1300
shared-tail watcher           = still waiting for step1500 decoded eval
GPU0 R2R worker               = running
GPU1 main Stage2              = running
```

step1500 仍未到达；当前是正常等待，不需要手动干预。

2026-08-17 11:05 状态更新：

```text
main Stage2 latest train step = 1400
main train/loss_visual        = 0.05621058866381645
main train/loss_pose          = 0.000789642333984375
main latest eval step         = 1400
main eval/loss_visual         = 0.07823165692389011
main eval/loss_pose           = 0.0014257431030273438
shared-tail watcher           = still waiting for step1500 decoded eval
```

`eval@1400` 与 `eval@1300` 基本持平，说明主 Stage2 generation + pose
co-train 仍稳定。距离 step1500 约 100 train step，按当前速度约 18–20 分钟。

2026-08-17 11:15：旧版 Stage One 命令收敛速度复核。

复核对象是旧版 `legacy20 + Wan2.1-1.3B official + latent_prefix +
history_iw_chunks=1,4,8,16 + full finetune + micro batch 1 +
grad_accum 16` 路径，入口为：

```bash
bash scripts/run_v1_wan_stage1_t4_history.sh cuda:0 <run_name> 1,4,8,16
```

当前短 probe 的 step1 与旧记录完全对齐：

```text
run = log/legacy20_prevcmd_convergence_probe_50step_20260817_110822/
step1 loss = 1.613098
history_iw_chunks = 16
history_micro_steps = 107
peak_reserved_gib = 31.289
```

该 probe 在 step1 后未继续占 GPU，因此不作为 50-step 结论；收敛速度仍以已
完成的同命令日志为准：

| run | step1 | step5 | step10 | step20 | wall-time / optimizer step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy20_stage1_t4_fullmix_convergence_20260817_021207` | 1.613098 | 0.875265 | 0.815112 | 0.572449 | 约 74.5s |
| `legacy20_stage1_prevcmd_20step_20260817_090828` | 1.613098 | 0.875129 | 0.814849 | - | 约 71.2s |

更早的同结构 1000-step 正式 run：

```text
run = log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/
step1    = 1.517262
step10   = 0.645557
step20   = 0.473429
step50   = 0.296863
step100  = 0.167988
step200  = 0.085224
step500  = 0.062960
step1000 = 0.053482
avg wall-time / optimizer step = 约 75.4s
```

结论：旧版生成支路的特征是 **前 20–200 optimizer steps 快速下降**，
1000 step 附近进入 `~0.05–0.07` 区间。这里的 optimizer step 是 effective
batch size 16，即 16 次 micro forward/backward 后做一次 optimizer update。
因此旧版“loss 收得快”和“单 optimizer step wall-time 约 1.2 分钟”同时成立。

2026-08-17 11:20：Stage2 RE10K action input 对齐问题与后续 schema 修正。

发现当前 `RE10KStage2Dataset` 虽然落盘 latent payload 中同时包含 `move`
与 `view`，但 Stage2 IW-aligned 路径只读取了 `move`，并把同一个序列同时作为
Wan native action encoder 的 `move/view` 输入。这与旧版有效 Stage One 命令不
一致；旧版训练明确分开读取并传入：

```text
history_move_batches -> condition_history_chunk(..., move, view)
history_view_batches -> condition_history_chunk(..., move, view)
move_batch           -> WanModel(move=move)
view_batch           -> WanModel(view=view)
```

该发现是有效的，但“分开 move/view 作为正式主 schema”随后被 supersede。
当前正式规则改为 **单一 trans-rot combo action**：

```text
combo_id = trans_id * 12 + rot_id
combo_dim = 144
```

样本只暴露一种 action；区别是用途：

```text
video datasets:
  action 用作 a_condition / A_hist / A_cur

VLN datasets:
  action 用作 a_label / A_out supervision
```

当前代码修正为：

```text
scripts/train_v1_stage2_re10k_pose_video.py
  dataset returns:
    a_hist_combo, a_cur_combo
  and keeps compatibility/audit fields:
    a_hist_move, a_hist_view
    a_cur_move,  a_cur_view
  and keeps old aliases:
    a_hist_primitives = a_hist_combo
    a_cur_primitives  = a_cur_combo

scripts/train_v1_stage3_iw_aligned_r2r_policy.py
  dataset returns:
    a_hist_combo
    action_combo
  policy supervision:
    CE(combo_logits, action_combo)

src/nav/v1/models/iw_aligned.py
  forward_core prefers a_hist_combo/a_cur_combo/action_combo.
  legacy_iw_move_view mode splits combo back into move/view only as IW native
  action_encoder shim.
  shared_tail_token mode encodes A_cur from combo.
  policy branch outputs combo_logits=[B,H,144].
```

旧 move/view split preflight：

```text
run = log/v1_stage2_iw_aligned_re10k_pose_video/preflight_move_view_split_20260817_111834/
dataset_size = 269
history_latents = [1, 16, 4, 56, 112]
a_hist_move     = [1, 10]
a_hist_view     = [1, 10]
a_cur_move      = [10]
a_cur_view      = [10]
z_future_noisy  = [16, 4, 56, 112]
pose_target     = [4, 9]
```

combo schema full-data preflight 通过：

```text
run = log/v1_stage2_iw_aligned_re10k_pose_video/preflight_combo_action_stage2_v2_20260817_160520/
dataset_size     = 269
history_latents  = [1, 16, 4, 56, 112]
a_hist_combo     = [1, 10]
a_cur_combo      = [10]
z_future_noisy   = [16, 4, 56, 112]
pose_target      = [4, 9]
```

Stage3 R2R schema read-through 通过：

```text
history_latents      = [4, 16, 1, 56, 112]
a_hist_combo         = [4, 10]
action_target        = [10, 6]   # compatibility continuous delta, not primary label
action_combo         = [10]
action_combo_minmax  = [3, 12]
```

注意：Stage3 full-model 1-step forward/backward 在 schema 更新后尚未重新跑完；
刚才被用户中断。因此当前只证明 schema/data read-through 与 Stage2 data
preflight，不能声称 Stage3 训练链路已重新验证。

当前已在跑的 2k Stage2 进程是在 combo schema 修正前启动的，因此它的结果仍属于
旧 action routing 版本，可继续作为历史诊断，但不能作为 combo schema 后的最终
Stage2 结论。

原本等待 step1500 后启动的 `shared_tail_token` watcher 已在 11:20 停止；
之后 `moveview_split_gate` watcher 也因 combo schema 更新停止，避免混变量实验：

```text
tmux session = nav_shared_tail_gate_after_step1500  # killed
tmux session = nav_moveview_split_gate_after_step1500  # killed
reason = action schema changed to combo; old move/view split gate no longer
         represents the intended single-variable experiment.
```

下一步 gate 顺序需要改为：

1. 在 combo schema 后重跑一个短 Stage2 gate：
   `fixed_rnull_unified + legacy_iw_move_view + a_hist/a_cur_combo`；
2. 通过后再只切 `current_action_mode=shared_tail_token`，验证 A_cur combo token
   routing；
3. 最后重跑 Stage3 full-model 1-step，再启动正式 policy 训练。

旧 `scripts/watch_step1500_then_run_shared_tail_gate.sh` 已加默认保护：
未显式设置 `ALLOW_SUPERSEDED_SHARED_TAIL_WATCHER=1` 时会直接退出，避免误跑
混变量 shared-tail 实验。

2026-08-17 19:02：启动 combo schema Stage2 cotrain gate。

目标：在“已经证明可收敛的旧 IW/Wan native generation branch + Stage2 pose
cotrain”上，只替换为当前正式 combo action 数据接口，验证是否仍接近旧版训练速度
和收敛速度。暂不切 `fixed_rnull_unified` / `shared_tail_token`，避免多变量混合。

```text
run = log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216/
tmux session = nav_combo_schema_stage2_gate_1k
tensorboard  = port 6026
device       = cuda:0
init         = Wan2.1-1.3B official weights
video branch = legacy IW/Wan native branch
register     = legacy_extract_update
A_cur route  = legacy_iw_move_view, but source field is a_cur_combo
steps        = 1000
batch        = 1
grad_accum   = 1
history      = 1 T4 chunk
lambda_pose  = 0.1
action loss  = 0.0
```

preflight 证明 Stage1/2 已包含第三阶段变量接口，但不监督 action 输出：

```text
history_latents = [1, 16, 4, 56, 112]
a_hist_combo    = [1, 10]
z_future_noisy  = [16, 4, 56, 112]
a_cur_combo     = [10]
a_noise         = [10, 6]
action_timestep = []
pose_target     = [4, 9]
combo_dim       = 144
```

早期 loss：

| step | train/loss_visual | train/loss_pose | seconds/step window | 对照 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.200185 | 0.431641 | 11.14 | 低于上一版 cotrain step1=1.406 |
| 10 | 0.546000 | 0.275391 | 9.09 | 低于旧 Stage1 step10=0.646，也低于上一版 cotrain step10=0.898 |
| 20 | 0.521377 | 0.120605 | 10.85 | 接近旧 Stage1 step20=0.473，略好于上一版 cotrain step20=0.546 |
| 30 | 0.469564 | 0.080078 | 11.05 | 已到旧 Stage1 step20 附近 |
| 40 | 0.397195 | 0.025146 | 11.12 | 继续下降，pose loss 明显下降 |
| 50 | 0.394404 | 0.011108 | 11.06 | 比旧 Stage1 step50=0.297 慢，但接近上一版 cotrain step50=0.380 |
| 60 | 0.404959 | 0.009155 | 11.11 | 有采样波动 |
| 70 | 0.322121 | 0.002731 | 11.03 | 继续下降 |
| 80 | 0.494771 | 0.003357 | 11.02 | 单点回弹 |
| 90 | 0.288870 | 0.002396 | 11.08 | 回到下降趋势 |
| 100 | 0.179318 | 0.003143 | 11.07 | 几乎贴近旧 Stage1 step100=0.168 |

step100 eval：

```text
eval/loss_visual = 0.5723168849945068
eval/loss_pose   = 0.0069732666015625
```

解释：train visual 在 step100 已经对齐旧版量级，但 eval@100 明显高于上一版
cotrain 的 `eval/loss_visual=0.313`。当前 eval 只取 4 batches，且
`RE10KStage2Dataset.__getitem__` 使用内部随机 window，因此单点 eval 方差较大；
不能只凭 step100 eval 判定失败。继续观察 step200/500/1000，并在后续正式 gate
中考虑固定 eval window 或扩大 eval batches。

当前只能说明 **早期下降健康且速度不慢**；完整验收仍需至少看 step100/200/500/1000
以及 eval/loss_visual、eval/loss_pose。若 step1000 eval visual 接近上一版
cotrain `~0.06` 和旧版训练 loss `~0.05–0.07`，才认为 combo schema 没破坏旧
生成支路收敛。

### Stage3 policy + video/pose replay gate（2026-08-17）

`scripts/train_v1_stage3_iw_aligned_r2r_policy.py` 已升级为支持可选 replay：

```text
policy batch:
  R2R rendered obs latent + instruction/action target
  -> L_action_flow + lambda_ce CE_aux
  -> backward

replay batch:
  RE10K T4 latent + pose
  -> L_visual + L_pose
  -> lambda_video_replay * L_visual + lambda_pose_replay * L_pose
  -> backward

optimizer.step()
```

这里 policy backward 和 replay backward 分开执行，避免同时保留两张大计算图。
`IWAlignedWorldNavModel.forward_stage2` 新增非 detach 的
`loss_visual_tensor/loss_pose_tensor`，专供 replay 反传；日志仍使用 detach 后的
`loss_visual/loss_pose`。

验证 run：

| 项 | 结果 |
| --- | --- |
| run | `log/v1_stage3_iw_aligned_r2r_policy/stage3_actionflow_with_replay_1step_20260817_043243/` |
| policy 数据 | rendered R2R H=10 smoke obs-latent cache |
| replay 数据 | RE10K T4 latent + pose |
| replay 系数 | `lambda_video_replay=0.25`，`lambda_pose_replay=0.05` |
| step1 policy | `loss_action_flow=1.2578`，`CE_aux=2.7656` |
| step1 replay | `loss_video_replay=0.7805`，`loss_pose_replay=0.5234`，weighted=`0.2213` |
| 速度/显存 | `24.74s/step`，`39.91GiB`（同时后台跑 R2R obs latent 编码） |

two-pass replay 回归（2026-08-17 更新后）：

| 项 | 结果 |
| --- | --- |
| run | `log/v1_stage3_iw_aligned_r2r_policy/stage3_twopass_replay_regression_1step_20260817_050513/` |
| policy 数据 | rendered R2R smoke obs latent；`history_latents=[4,16,1,56,112]` 与 `A_hist=[4,10]` 仅用于 Register 预处理 / update；policy 侧读取 `Register + Z_obs + instruction/text + A_noise`，不直接读取 `A_hist` 或原始 `Z_history` |
| replay 数据 | RE10K T4 latent + pose；replay video pass 保持 strict IW layout，pose 走 separate current-clean probe |
| step1 policy | `loss_action_flow=2.3281`，`CE_aux=2.7813` |
| step1 replay | `loss_video_replay=1.0281`，`loss_pose_replay=0.5508`，weighted=`0.2846` |
| 速度/显存 | `23.57s/step`，`28.12GiB` |

Stage3 dataset reader 也已支持把 `--obs-latent-manifest` 指向一个 full cache run
目录；脚本会自动读取 `manifests/encoded_episodes*.jsonl(.gz)`，因此 full cache
完成后不需要手动合并 sharded manifest。

Stage3 数据入口进展（2026-08-17）：

```text
raw H=10:
  /sharedata/NAV/derived/v1/vln/raw_policy_h10/20260817_034014/

rendered H=10:
  /sharedata/NAV/derived/v1/vln/rendered_policy_h10_r2r/20260817_034242/
```

统计：

| source | policy chunks |
| --- | ---: |
| R2R-CE standard train | 631,244 |
| R2R-CE standard val_seen | 46,158 |
| R2R-CE standard val_unseen | 105,459 |
| total | 782,861 |

该 manifest 已完成 `--check-files`：`skipped_missing_frame=0`。每条 row 包含
`obs_frame_path`、`history_frame_paths`、`instruction`、`previous_action_ids`、
`target_action_chunk[horizon=10]`。下一步若训练 policy，需要先把 RGB observation
编码为 VAE latent 或构建在线/离线 latent cache。

| 实验 | run | 端口 | 关键设置 | 结果 |
| --- | --- | --- | --- | --- |
| `rflowfix_fullmodel` | `log/v1_stage2_re10k_pose_video/v1_stage2_rflowfix_mlp_lam0p1_re10k_300step_20260817_003709/` | `6018` | 仍用 `V1SharedWanBackbone`，只修 RFlow target / timestep 混合 | step10 `loss_visual≈1.97`，未回到旧量级，说明只修公式不够 |
| `iw_aligned_current_prefix_pose` | `log/v1_stage2_iw_aligned_re10k_pose_video/v1_stage2_iw_aligned_fastdrop_30step_20260817_012257/` | `6019` | 原生 WanModel；825 key；`z_obs` full current prefix；`λ_pose=0.1` | 6–7s/step，峰值约 40.2G；pose eval≈0.022；video eval≈1.25 |
| `iw_aligned_cached_text` | `log/v1_stage2_iw_aligned_re10k_pose_video/v1_stage2_iw_aligned_text_fastdrop_15step_20260817_012814/` | `6020` | 同上，但使用 cached UMT5 empty text，而非 zero text | video eval≈1.27，说明 zero text 不是主因 |
| `iw_aligned_strict_videoonly` | `log/v1_stage2_iw_aligned_re10k_pose_video/v1_stage2_iw_aligned_strict_videoonly_15step_20260817_013159/` | `6021` | 严格 IW video branch：`image_cond=Register`，`local_memory=z_obs` latest frame，`λ_pose=0` | 约 4.1s/step，峰值约 27.9G；video eval≈1.27 |
| `iw_aligned_updatecurrent_videoonly` | `log/v1_stage2_iw_aligned_re10k_pose_video/v1_stage2_iw_aligned_updatecurrent_videoonly_15step_20260817_013429/` | `6022` | strict video-only，再把 `z_obs/current` 也 update 到 register，用于定位旧 IW history 语义 | video eval≈1.27，说明 current 是否进入 register 不是短测主因 |

关键代码改动：

- `src/nav/v1/models/iw_aligned.py`：新增 IW/Wan 原生 generation branch 对齐模型。
- `scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py`：新增 RE10K Stage2/IW-aligned 训练入口，使用 `.venv_infinite_world` 环境、flash-attn、non-reentrant checkpoint。
- `../Infinite-World/infworld/models/dit_model.py`：只新增 `return_prefix_video_hidden`，用于 Stage2 probe 读取 clean prefix hidden；不改变 Wan 主计算。

下一步应做的不是继续微调这个 15-step ebs1 诊断，而是跑一个真正可比的对齐实验：

```text
数据: 旧 fullmix 数据配比 / 或至少与旧 run 相同 manifest
有效 batch: ebs16
窗口: 对齐旧 StageOne 的 history/current/target 构造
模型: IWAlignedWorldNavModel strict video branch
目标: 先复现旧 visual-only loss 曲线，再在同一设置上加 Stage2 pose probe
```

## 当前 Stage Two 对照实验（2026-08-15）

| 组别 | 状态 | GPU | run | 关键差异 | 观察用途 |
| --- | --- | --- | --- | --- | --- |
| cotrain_mlp_lam0p1 | Running | GPU0 | `log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648/` | `pose_head=mlp`，`lambda_pose=0.1` | 原始 Stage2 pose+video cotrain |
| video_only | Stopped / Diagnostic done | GPU1 | `log/v1_stage2_re10k_pose_video/v1_stage2_re10k_videoonly_sharedwan_wanfull_2k_20260815_025913/` | `lambda_pose=0.0` | 已发现 `loss_visual` 与原 cotrain 几乎重合，说明原 pose loss 更像 probe，不明显 reshape generation backbone |
| cotrain_linear_lam1 | Running | GPU1 | `log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_linear_lam1_sharedwan_wanfull_2k_20260815_143827/` | `pose_head=linear`，`lambda_pose=1.0` | 检查弱 pose head + 更强 pose loss 是否能更明显影响 shared WanBlock / video loss |

TensorBoard 对比端口：`6017`，当前只包含 `cotrain_mlp_lam0p1` 与 `cotrain_linear_lam1`。

## 论文实验实现时间线（Implementation Roadmap）

本节从 `essay/iclr2027/iclr2027/main.tex` 的论文路线反推工程任务。当前会话负责按
此顺序实现训练、评测、对比和 ablation；所有正式结论必须同时回写 `doc/`、
`log/` 与 `result/`。允许缩小 steps / batch / subset 做 formal diagnostic，
但不得使用 scaffold / toy model 作为验收证据。

### 总目标

论文主问题不是“新做一个 VLN policy”，而是验证：

```text
3D-aware streaming world model 的 fixed-budget Register memory
是否能成为 embodied navigation 的 observation history / world representation。
```

因此任务顺序必须同时服务三条实验轴：

| 论文问题 | 工程目标 | 最终产物 |
| --- | --- | --- |
| Q1: Register 是否 transfer 到 VLN？ | Stage3 policy training + VLN eval | `result/vln_transfer/<run>/` |
| Q2: Register 是否是更好的 streaming world-model memory？ | long-horizon generation 对比 InfiniteWorld / HPMC / window | `result/generation_memory/<run>/` |
| Q3: 其他 memory/representation 是否同样可迁移？ | KV/cache/window/map/RSSM/VGGT 等 adapter 对比 | `result/memory_transfer/<run>/` |

### 时间顺序任务列表

| 顺序 | 阶段 | 类型 | 任务 | 关键对照 / ablation | 验收产物 |
| ---: | --- | --- | --- | --- | --- |
| 0 | 基础检查 | Infrastructure | 冻结当前 formal V1 结构：`V1SharedWanBackbone`、`RegisterCell`、`A_hist/A_cur/A_noise`、policy-safe mask；确认所有脚本不再走 dual-stream/scaffold | `smoke_v1_full_pipeline.py` 必须用完整模型结构 | `log/full_pipeline_smoke/<date>/report.json` |
| 1 | 数据闭环 | Data | 完成 Stage1/2/3 所需数据索引：T4 video latent、pose/action schema、VLN rendered obs/action chunk；统一 manifest 与 dataset reader | 数据读取顺序 shuffle；不污染原始下载目录 | `/sharedata/NAV/derived/v1/...` + `02_data/data_preparation_schema_and_status.md` |
| 2 | Stage1 core | Core training | 训练 predictive video memory：DL3DV + SpatialVID20 + RE10K + Argoverse2，history 覆盖 IW-1/4/8/16 等价长度；action tokens 存在但 action loss=0 | no Register、short-only history、local-only current obs | `log/stage1_v1/<run>/`，`result/generation_memory/stage1_probe/<run>/` |
| 3 | Stage1 generation eval | Core evaluation | 评估 long-horizon generation memory：Register vs local-only / sliding window / InfiniteWorld-HPMC / full-context oracle | 固定 latency / token budget / history span | FVD/LPIPS/PSNR/SSIM、pose consistency、latency、GPU memory |
| 4 | Stage2 diagnostic closeout | Core training | 完成当前 RE10K pose-only formal 对照：`mlp λ=0.1` 与 `linear λ=1.0`；判断 pose loss 是否 reshape shared backbone | video-only 已停止作为历史诊断；继续比较 `loss_visual`、`loss_pose`、hidden probe | `log/v1_stage2_re10k_pose_video/...`，TensorBoard `6017` |
| 5 | Stage2 design selection | Decision | 根据 Stage2 diagnostic 选择正式 3D head/loss：linear probe / weak MLP / multi-layer probe / register-after-backbone probe；冻结 `λ_pose` 初值与 warmup 策略 | `λ_pose∈{0.1,0.5,1.0}`，pose head capacity，probe layer | `decision_log.md` 新 DEC + 训练文档更新 |
| 6 | Stage2 core | Core training | 使用 Stage1 checkpoint 启动正式 Stage2：保持 `L_visual`，加入 pose/depth/geometry supervision；优先 RE10K + DL3DV pose，随后加 VGGT pseudo labels | no spatial loss、pose-only、depth-only、register-hidden vs obs-hidden supervision | `log/stage2_v1/<run>/`，`result/geometry_probe/<run>/` |
| 7 | Stage2 probe eval | Core evaluation | 评估 spatial readability：pose error、relative pose、depth/point probe、layer selection、Register vs current hidden | frozen probe vs finetuned probe；不同 layer / token slice | `result/geometry_probe/<run>/metrics.json` |
| 8 | Stage3 data readiness | Data | 完成 R2R-CE / RxR-CE / ScaleVLN / LHPR-VLN 的 rendered obs、instruction、action chunk、STOP label；构造 LeRobot-like shard/index | 先 R2R-CE formal subset，再扩全量 | `result/data_audit/vln/<run>/` |
| 9 | Stage3 policy smoke | Core training | 用 Stage2 checkpoint 启动完整 Stage3 forward/backward：policy action-flow + STOP/CE + Stage1/2 replay；推理跑 policy-only | no scaffold；generation branch 可训练 replay，推理裁掉 | `log/stage3_v1_smoke/<run>/report.json` |
| 10 | Stage3 core training | Core training | 正式 VLN policy training：Register + current obs + instruction -> `H_nav=10` action chunk；闭环执行第一步后更新 Register | frozen backbone、frozen Register、full finetune、no replay | `log/stage3_v1/<run>/` |
| 11 | Q1 VLN transfer eval | Main experiment | 在 R2R-CE / RxR-CE 上评估 SR/SPL/NE/OSR/nDTW/Stop accuracy，按 seen/unseen、path length、history dependence 分层 | current-only、finite window、random Register、video-only Register、3D-aware Register | `result/vln_transfer/<run>/metrics.json` |
| 12 | Q2 generation memory paper table | Main experiment | 复现/对齐 InfiniteWorld-HPMC 与 NAV Register 的长程生成质量-效率曲线 | local-only、sliding window、HPMC、Register、full-context oracle | `result/generation_memory/<run>/summary.md` |
| 13 | Q3 alternative memory adapters | Main experiment | 适配其他 memory / representation 到 VLN：KV/cache、compressed window、semantic/spatial map、GRU/RSSM、VGGT feature、diffusion hidden without Register | matched token budget / matched latency / matched training data | `result/memory_transfer/<run>/` |
| 14 | Ablation batch A | Ablation | Register 机制消融：`K∈{32,64,128,256}`、no Register、random/frozen Register、history shuffle、early landmark removal | 主要看 long-horizon SR、generation consistency、memory budget curve | `result/ablation/register/<run>/` |
| 15 | Ablation batch B | Ablation | Training signal 消融：no `L_visual` pretraining、no 3D loss、no policy replay、pose-only vs depth+pose、linear vs MLP pose head | 主要验证 Stage1/2/3 每个 loss 是否必要 | `result/ablation/loss/<run>/` |
| 16 | Ablation batch C | Ablation | Action/mask 因果消融：remove `A_hist`、remove `A_cur`、remove `A_noise`、mask leakage test、current chunk double-write | 验证 `A_hist/A_cur/A_noise` 分离和 policy-safe mask 的必要性 | `result/ablation/action_mask/<run>/` |
| 17 | 论文表格冻结 | Paper | 把 Q1/Q2/Q3 + ablation 的 verified 结果写入 `essay/iclr2027/iclr2027/main.tex`，未跑完的一律保留 reserved/TBD | 概念性结果必须标 `Verified/Running/Unverified` | essay 表格、bib、result pointer |

### 优先级与停止条件

1. **最高优先级**：Stage2 是否真的 reshape representation。若 `linear λ=1.0`
   仍只让 pose head 学动而 `loss_visual`/hidden probe 不变，则 Stage2 要改为
   multi-layer / Register-hidden / weaker probe 或加入 depth/point supervision。
2. **第二优先级**：Stage3 policy smoke。只要 Stage2 有一个可用 checkpoint，就应尽快
   打通完整 VLN action-flow 训练与 policy-only inference，避免世界模型侧越训越远但
   无法服务导航。
3. **第三优先级**：Q2 long-horizon generation。它是证明 Register memory 不是普通
   cache 的关键，但必须与 Q1 VLN transfer 配对，否则文章会退化成视频生成 memory 论文。
4. **停止/转向条件**：若 Register 对 VLN 不优于 finite window / KV cache，则论文主线应从
   “Register better” 转为 “world-model memory transfer diagnostic”，并强化失败分析。

### 最小可投稿结果包

若时间不足，最低限度需要完成：

```text
Q1:
  current-only / finite-window / video-only Register / 3D-aware Register / NAV full

Q2:
  local-only / InfiniteWorld-HPMC / NAV Register 的同预算 generation-memory 曲线

Ablation:
  no Register
  no spatial loss
  frozen vs finetuned Register
  mask leakage / no A_cur
```

没有 Q1 VLN transfer 结果时，不应把文章写成 navigation paper；没有 Q2 generation-memory
对比时，不应声称 Register 是更好的 streaming world-model memory。

## 合并主题

- V1 三阶段训练数据、组织与 Loss 设计
- V1 Stage One T4-IW 对齐训练规范
- V1 Stage Two RE10K Pose-only 正式验证训练
- V1 Stage Two RE10K 视频生成与 3D 监督 Pilot
- V0 Register 流式训练样本与在线状态语义
- V0 Register 流式世界模型 RE10K 数据与训练记录
- V0 Stage One DL3DV 从头训练实验
- V0 Stage Two 视频生成前向中的 3D Probe 与监督规范
- V0 多数据集增量预处理与分阶段训练

## 维护规则

- 后续相关主题优先更新本文档，不再新增细碎同主题文档。
- 历史 run、旧结构和 superseded 方案保留在对应章节中，用于追溯和对照。
- 若代码/日志/文档三线不一致，以代码与真实记录为准先修正文档。

## 合并正文


## V1 三阶段训练数据、组织与 Loss 设计


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| ID | NAV-TRN-006 |
| 类型 | 训练数据与损失设计 |
| 状态 | Accepted Draft / V1 Final Training Skeleton |
| 更新时间 | 2026-08-14 |
| 职责 | 定义 V1 GigaWorld-like + Register NAV 的 Stage One/Two/Three 训练数据来源、样本组织、缓存格式、loss 设计和数据配比 |

### 核心结论

V1 不再围绕 81-frame dense video chunk 构造训练，而围绕 Register memory、
action-compatible interface 和 sparse future consequence 构造：

```text
history/current obs/instruction
  -> Register + fixed action-compatible tokens
  -> generation branch 预测 sparse future consequence
  -> Stage Three policy branch 预测 action/control
```

三阶段数据职责如下：

| 阶段 | 主目标 | 主要数据 | 推理保留 |
| --- | --- | --- | --- |
| Stage One | 让 shared DiT/Register 学会长历史视频记忆和 future consequence；action 接口固定存在，但 action output 不训练成 policy | DL3DV、SpatialVID、RE10K、Argoverse2 | generation；action interface present but not policy |
| Stage Two | 在同一视频生成前向中加入 3D supervision，找可用于导航的 Register/hidden layer | DL3DV、RE10K、SpatialVID、Argoverse2，配合 VGGT/pose pseudo label | generation + 3D probe；action interface fixed |
| Stage Three | 用 VLN 数据训练 policy/action output，同时 replay Stage One/Two 数据与 loss 防止 shared backbone/Register 退化 | R2R-CE、RxR-CE、ScaleVLN、LHPR-VLN + Stage One/Two replay pool | 训练 cotrain；推理 policy-only，generation branch 删除 |

V1 需要新建独立派生目录，不覆盖 V0：

```text
/sharedata/NAV/derived/v1/
  manifests/
  frame_packs/
  vae_packs/
  actions/
  geometry/
  vln/
  audits/
```

旧 `full_episodes_v1` 81-frame latent 属于 V0 dense chunk 口径，不作为 V1 主缓存。
早期 V1 sparse pack `[obs, future_1, future_2, future_3, future_4]` 已完成
smoke，但主线训练改为 `T_latent=4` micro chunk；具体规则见
`training_plan_and_experiment_log.md`。

一个必须遵守的约束是：**pose 不是模型推理输入**。Pose 只在训练数据准备和
监督中使用：

```text
pose 可用于:
  pseudo action / displacement bucket 构造
  3D supervision target
  数据审计和尺度校准

pose 不可用于:
  Stage Three policy condition
  真实世界推理必需输入
  无 pose benchmark 的必需输入
```

V1 的 history/Register 因此不是 `pose-conditioned memory`，也不是把当前 GT action
塞进 policy condition。它应是：

```text
implicit Register storage
previous-action/motion-aware update via A_hist tokens when available
pose-supervised when available
```

真实世界中没有 GT pose 是正常情况；但 agent/robot 通常知道自己执行过的
action command，可能还可选获得 odometry/IMU。V1 默认允许历史已执行 action/motion
以 `A_hist` tokens 参与 Register update，但禁止 action bias / latent bias 注入。
当前要预测的 target action 不应在 Stage One 被强行训成 policy；它在视频生成数据中
最多作为 generation condition 或格式占位。真正的 policy/action output 在
Stage Three 用 VLN 数据训练。

### 统一样本结构

Action、GeometryTarget、VaePack 和 V1Sample 的统一字段定义见
`../02_data/data_preparation_schema_and_status.md`。本文件只描述三阶段如何使用这些
schema 训练。

### History / Current / Target 时间边界

V1 正式样本必须采用下面的流式边界，避免 latest current chunk 同时进入
Register 和 Local：

```text
给定连续 micro chunks:
  C_0, C_1, ..., C_{t-2}, C_{t-1}, C_t

用于预测 C_t 时:
  Register history = C_0 ... C_{t-2}
  Current / Local  = C_{t-1}
  Target future    = C_t
```

也就是：

```text
R_{t-2} = RegisterCell(... RegisterCell(R_null, C_0, A_hist0) ..., C_{t-2}, A_hist_{t-2})
z_obs   = latent(C_{t-1})
z_future_target = latent(C_t)
```

禁止把 `C_{t-1}` 既写进 Register 又作为 `z_obs/current` 输入。Register 承担
更早历史压缩，Current/Local 承担最近观测细节。

这个边界也决定 Stage Two 的 3D supervision：主要监督窗口放在
`C_{t-1}` current chunk 上，而不是放在被 Register 压缩后的老历史上。理由是
current chunk 保留了最完整的空间 token/grid hidden，最接近 VGGT-Ω 中每帧
Patch Tokens + Camera/Register Tokens 被 camera/depth heads 读取的结构。

2026-08-14 曾按该口径运行一次 RE10K Stage Two diagnostic，但该 run 缩小了
latent spatial resolution 与 backbone depth，**不作为正式训练、正式 smoke 或验收
证据**。记录仅用于追溯错误口径：

```text
NAV/doc/03_training/training_plan_and_experiment_log.md
NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/
```

### 按数据集确定的训练变量表

以下表格是数据构建和 dataloader 的准绳。`EMPTY` 表示该变量以空 embedding /
unknown embedding / zero mask 的确定形式进入，不再称为“可选”；`不参与` 表示该
数据集不进入该阶段或该变量不构建。

#### Stage One / Stage Two：视频数据变量

| 数据集 | Stage One | Stage Two | `C_hist` 历史视频 chunk | `A_hist` 历史 action/motion | `C_local` 当前/local chunk | `A_cur` 视频生成当前 action condition | `A_noise` action 主变量 | `Z_future` 视频 target | `Text` | `G_3D` 三维监督 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DL3DV | 使用 | 使用 | T4 micro latent windows | 由 GT pose 差分生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 pose 差分 pseudo motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | GT pose/intrinsic 生成 relative pose/depth/point target |
| SpatialVID 20% | 使用 | 使用 | T4 micro latent windows | 使用落盘 action/motion 生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 action/motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | 使用已有稀疏 pose 或 VGGT pseudo geometry，带 confidence mask |
| RE10K | 使用 | 使用 | T4 micro latent windows | 由相机 pose 差分生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 pose 差分 pseudo motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | camera pose/intrinsic 生成 relative pose/depth/point target |
| Argoverse2 | 使用 | 使用 | T4 micro latent windows | 由 ego pose / vehicle motion 生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 ego-motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | metric ego pose / camera calibration 生成 3D target |
| Kinetics | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 |

Stage One 的固定 loss 规则：

```text
L_stage1 = L_visual_flow
action output branch:
  A_cur 必须进入 DiT condition/context；
  A_noise tokens 必须进入 single shared WanBlock token stream；
  A_out 从 shared WanBlock 中的 action-token hidden 经 Giga/DreamZero-style
  action flow decoder 读出；
  action loss = 0；
  不把视频 pseudo action 当 policy supervision。
```

Stage Two 在 Stage One forward 基础上增加：

```text
L_stage2 = L_visual_flow + λ_3d L_3D
```

#### Stage Two current-chunk 3D 监督

参考 VGGT-Ω 的设计，Stage Two 不只监督一个全局 register 向量，而是在 shared
backbone 的若干候选层读取：

```text
current visual hidden tokens     # 对应 C_{t-1} 的 patch/latent-grid tokens
register-after-backbone tokens   # 历史上下文汇聚后的 scene/memory tokens
```

推荐第一版 probe：

```text
current hidden grid
  -> DepthHead / PointHead
  -> depth / point_map / confidence

register + pooled current hidden
  -> Camera/PoseHead
  -> relative pose / camera motion
```

其中 dense depth/point supervision 只作用在 current chunk 的可见视角上；
relative pose/camera supervision 可使用 current chunk 内帧间 pose，或
`C_{t-1} -> C_t` 的短期相机运动。Pose/depth/point 仍然只作为训练监督，不作为
推理输入。

Stage Two dataloader 因此需要为每个样本额外构造：

```text
geometry_window = C_{t-1}
geometry_target:
  intrinsics_current
  poses_current
  relative_poses_current
  depth_current or point_map_current
  confidence / valid_mask
  teacher_source
```

可用 teacher 来源按优先级：

```text
1. simulator / GT pose + depth
2. dataset camera pose + sparse/interpolated depth/point target
3. VGGT/VGGT-Ω pseudo depth / point map / confidence
```

Loss 第一版：

```text
L_3D =
  λ_depth * masked_scale_shift_depth_loss(D_pred, D_gt)
+ λ_point * masked_point_l1_or_l2(P_pred, P_gt)
+ λ_pose  * relative_pose_loss(T_pred, T_gt)
+ λ_conf  * confidence_calibration_loss(optional)
```

VGGT-Ω 给我们的直接启发是：保留 dense current tokens 做 3D readout，同时用
Register/scene tokens 提供跨视角上下文；不要只用一个 compressed Register 去
恢复全部空间细节。

#### Stage Three：VLN 数据变量

| 数据集 | Stage Three | `C_hist` 历史观测 | `A_hist` 历史 action（仅 Register update） | `C_local` 当前观测 | `Instruction/Text` | `A_cur` 视频生成当前 action condition | `A_noise` action 输入 | `A_out` 监督 | `Z_future` 视频 target | `G_3D` 三维监督 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R-CE | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | R2R instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| RxR-CE | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | RxR instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| ScaleVLN | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | ScaleVLN instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| LHPR-VLN | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | LHPR instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| MP3D scene assets | 场景资源 | 渲染来源 | 不单独提供 | 渲染来源 | 不单独提供 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 |

Stage Three 的固定 policy batch 规则：

```text
VLN policy batch:
  L_nav = L_action_flow(A_noise -> A_out, gt_actions)
  可附加 discrete CE / STOP balance auxiliary，但不作为唯一主结构
  不构建 Z_future，不把 future visual/3D 作为 policy condition。

Replay batch:
  从 Stage One/Two 数据池采样 video/3D windows；
  继续计算 L_visual_flow / L_3D；
  用于保持 shared backbone/Register 的 video memory 与空间表征。
```

Stage Three 总 loss：

```text
L_stage3 =
  L_action_flow
+ optional λ_ce L_discrete_ce_aux / λ_stop L_stop_balance
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d   L_3D_replay
```

推理时：

```text
只执行 policy/action path；
generation/video/3D heads 不执行。
```

#### Video/action 样本

每个 video 样本统一为：

```text
episode_id
dataset_name
current_frame_index t
history_indices h_0...h_m
future_sparse_indices f_1...f_4
pose/intrinsic for selected frames, if available
pseudo_action a_{t:t+H_action-1}, H_action=10
text/caption or empty prompt
scale metadata
confidence metadata
```

进入模型：

```text
Register:
  R_{-1} = fixed R_null
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  no action/latent/additive bias

shared WanBlock token stream:
  [Register tokens]
  [Z_obs clean]
  [Z_future noisy]
  [A_noise]  # action chunk 的待生成/去噪变量，不是 policy condition

condition:
  text / timestep / modality / dataset / action-scale
```

其中 V1 generation branch 当前主线固定：

```text
RGB micro chunk:
  13 frames, stride 12 with adjacent chunks

VAE latent:
  history C_i and target C_target

shape:
  C_i      = [B,16,4,H_lat,W_lat]
  C_target = [B,16,4,H_lat,W_lat]
```

`T_latent=4` 对应13帧短期 future consequence。长程能力通过 Register 覆盖的
history/context span 与 InfiniteWorld 对齐，而不是通过增大 noisy target T。

#### VLN policy 样本

每个 VLN 样本统一为：

```text
dataset_name
episode_id
scan/scene_id
instruction
step index s
current observation obs_s
history observation/action prefix
target low-level actions a_s...a_{s+H_nav-1}, H_nav=10
future observations: 不构建，第一版 Stage Three 不启用 generation auxiliary
pose/depth/goal distance: 不作为输入；第一版不构建 3D/progress auxiliary
```

Stage Three 推理时：

```text
generation branch:
  disabled, T_future = 0

policy input:
  obs_s + instruction + Register + A_noise tokens

policy output:
  action velocity/noise -> denoise/decode -> action chunk [B,H_nav]
```

默认：

```text
H_nav = 10
执行策略 = first-action execution
```

也就是模型预测 10 步，但闭环只先执行第 1 步，再用新 observation 更新 Register。

### V1 数据预处理与组织

#### Step 1：Source manifest

所有数据先生成 source manifest：

```text
/sharedata/NAV/derived/v1/manifests/source_<dataset>.jsonl
```

每行包含：

```json
{
  "dataset": "...",
  "episode_id": "...",
  "frame_paths": ["..."],
  "poses": "... or inline summary",
  "intrinsics": "...",
  "fps": 30.0,
  "has_metric_scale": true,
  "has_instruction": false,
  "split": "train"
}
```

#### Step 2：Action / displacement 标注

视频数据不按固定帧数切，而按空间位移 bucket 构造 future：

```text
small:
  约 1 个 Habitat forward，≈0.25m 或 episode-normalized 等价位移

medium:
  约 2-4 个 Habitat forward，≈0.5-1.0m 或等价位移

rotation:
  约 30° / 60° / 90°
```

有 metric pose 的数据：

```text
使用 meter/radian
```

无可靠 metric scale 的数据：

```text
使用 episode median motion 归一化
```

无 pose 数据：

```text
优先尝试 VGGT / optical flow / inverse dynamics 生成 pseudo motion；
若 confidence 不足，则不构造 motion-aligned history 样本。
```

action 标注输出：

```text
/sharedata/NAV/derived/v1/actions/<dataset>_actions.jsonl
```

#### Step 3：Sparse frame pack manifest

为 Stage One/Two 建立 sample manifest：

```text
/sharedata/NAV/derived/v1/manifests/stage1_video_packs.jsonl
/sharedata/NAV/derived/v1/manifests/stage2_3d_packs.jsonl
```

每个 sample 明确记录：

```text
obs frame
4 个 future sparse frames
history rollout frames
action horizon
displacement bucket
dataset weight
confidence
```

#### Step 4：VAE pack cache

V1 不以 81-frame chunk 为主缓存，而缓存 5-frame sparse pack：

```text
/sharedata/NAV/derived/v1/vae_packs/<dataset>/<sample_id>.pt
```

建议字段：

```python
{
  "z_obs": Tensor[16,1,H_lat,W_lat],
  "z_future": Tensor[16,1,H_lat,W_lat],
  "pack_frame_indices": [t, f1, f2, f3, f4],
  "actions": Tensor[10, action_dim_or_labels],
  "poses": ...,
  "intrinsics": ...,
  "scale": ...,
  "confidence": ...
}
```

训练初期可 on-the-fly VAE encode 做 smoke；正式训练应缓存 pack latent，避免每步
重复跑 VAE。

#### 无 pose / 无 action 数据的降级路径

无 pose 或无 action 的视频不能作为主 Register history 训练样本，因为模型无法知道
观测变化由什么 motion 造成。它们可以按置信度分三类使用：

| 数据状态 | 使用方式 | 是否参与 Register update 主损失 |
| --- | --- | --- |
| 有真实 action/odometry 或可靠 pose | 主训练样本，监督 action + generation + 3D | 是 |
| 无 GT pose，但 VGGT/flow/inverse dynamics 产生可靠 pseudo motion | 低权重样本，带 confidence mask | 可少量参与 |
| 无 pose、无 action、无可靠 pseudo motion | 只做弱 generation/caption/domain regularization | 否 |

对于 generation benchmark 如果没有 action/pose：

```text
action token = null / no-op / text-implied motion
Register update = 不作为空间历史能力评估
指标用途 = 只看视觉质量或通用生成能力，不证明导航式 memory
```

这类 benchmark 可以跑，但不能用来证明 V1 的 action-stream-conditioned generation
或 spatial memory。

#### Step 5：3D pseudo label cache

Stage Two 需要：

```text
/sharedata/NAV/derived/v1/geometry/<dataset>/<sample_id>.pt
```

可包含：

```text
depth / point map / relative pose / camera trajectory / confidence mask
```

来源优先级：

```text
1. 数据集真实 pose/intrinsic
2. 可由 pose 多视角推导的 relative geometry
3. VGGT pseudo depth/point/camera，带 confidence mask
```

真实 pose、VGGT camera、depth 或 point map 只作为 teacher signal。Stage Two forward
接口不能把 GT pose 当作 condition 输入给 policy/action tokens。

### Stage One：Video generation memory pretraining

#### 目标

Stage One 训练 shared DiT/Register 的视频记忆与 consequence generation 能力：

```text
obs/history/Register/dataset-defined current action condition/text
  -> future visual consequence
```

Stage One 不是 policy 训练，也不把视频 pseudo action 当作有导航语义的 action
supervision。视频数据通常没有 instruction，很多数据也没有真实 embodied action；
因此 Stage One 的 action 只承担两个弱角色：

```text
1. history action / motion token:
  按上表确定：有 pose/action 的数据生成 previous motion/action；
  以独立 `A_hist` tokens 参与 RegisterCell 更新；
  禁止 action bias / latent bias 注入。

2. current action condition for video generation:
  按上表确定：有 pose/action 的数据生成 current action condition；没有则用 EMPTY。

3. noised action latent / action output:
   Stage One 只保留 token slot 和 forward 格式，不作为主要训练目标；
   默认不计算 policy/action loss。
```

真正有 policy 意义的 action output 在 Stage Three 用 VLN/R2R/RxR action label 训练。

#### 使用数据

推荐第一版数据：

| 数据 | 用途 | 备注 |
| --- | --- | --- |
| DL3DV | 高质量 pose、长 episode、主几何视频数据 | 权重大于其原始条数占比 |
| SpatialVID | 大规模短中视频、已有 action 标注 | 需要控制占比，避免分布压倒其他数据 |
| RE10K | 室内/房产相机运动、pose 可用 | 规模小但与 VGGT/3D 表征相关 |
| Argoverse2 | 自驾视角、metric ego motion | 当前条数少，作为 domain diversity |

Kinetics 暂不进入 Stage One 主训练；除非后续 VGGT/action 置信度通过审计。

#### 样本组织

每个样本：

```text
history:
  若干过去 video chunks
  dataset-defined A_hist tokens
  从 fixed R_null 开始，用同一个 RegisterCell teacher-forcing 在线 roll Register

current:
  local/current video chunk -> local memory

future:
  下一个 T4 micro chunk -> Z_future target

action:
  dataset-defined current action condition for generation
  A_noise token placeholder, no policy loss by default
```

当前 T4-IW 对齐主线使用 window-level temporal samples：

```text
T4 micro chunk:
  latent T=4, RGB 13 frames, stride 12 frames

history_iw_chunks:
  IW-1  -> 7   T4 micro history updates
  IW-4  -> 27  T4 micro history updates
  IW-8  -> 54  T4 micro history updates
  IW-16 -> 107 T4 micro history updates

target:
  next T4 micro chunk
```

训练样本单位不是 episode，而是 temporal window：

```text
(episode_latent_path, history_iw, start_micro)
```

这使长视频数据可展开为大量 windows；Stage One 的“数据量”按 window-level 统计，
不按 episode 条数统计。

#### Loss

Stage One loss：

```text
L_stage1 =
  λ_v      L_visual_flow
+ λ_fmt    L_action_format
+ λ_reg    L_register_consistency
+ λ_aux    L_dataset_balance
```

建议第一版：

```text
L_visual_flow:
  只监督 Z_future 的 velocity/noise，不监督 clean Z_obs。

L_action_format:
  当前主实验固定 loss 权重为 0。该项只用于检查 action token slot / mask / head shape 是否能跑通；
  不把视频 pseudo action 当作真正 policy supervision。

L_register_consistency:
  第一版固定为 0；若后续打开需新建 ablation，不混入当前主实验。
```

推荐初始权重：

```text
λ_v = 1.0
λ_fmt = 0.0 initially
λ_reg = 0.0
```

只有在需要做接口 smoke 时，才把 `λ_fmt` 临时设为极小值，例如 `0.01`；
正式 Stage One 不以 action loss 作为优化目标。

#### 数据配比

不要按 episode 条数自然采样。Stage One 应按 window-level 和 history bucket
统计采样权重；SpatialVID episode 多但短，DL3DV episode 少但长，window-level
会自然改善比例。

```text
Short history / IW-1, IW-4:
  SpatialVID 可以占较高比例，但需要 cap。

Long history / IW-8, IW-16:
  DL3DV 应主导，因为它提供长视频和高质量 pose。
```

第一版以实际 window count 为基础，再做 dataset-aware cap；不再预设固定
`35/40/15/10` 条数比例。

### Stage Two：3D-supervised generation training

#### 目标

Stage Two 不改变 Stage One 的基本前向，只是在同一个 video generation forward
上额外增加 3D supervision 和 probe：

```text
same V1 generation-style forward
  + Stage One visual flow loss
  + selected hidden/Register layer
  + 3D heads/probes
```

核心不是另训一个 3D backbone，而是在不破坏 Stage One generation/Register memory
的前提下，确认 DiT/Register 的哪一层有可读 3D 表征。

#### 使用数据

Stage Two 优先使用几何可信数据：

| 数据 | 用途 | 建议权重 |
| --- | --- | ---: |
| DL3DV | 主 3D 数据，pose/intrinsic 质量最好 | 50% |
| RE10K | 与 VGGT/房产相机运动相关，适合相机几何 | 20% |
| SpatialVID | 大规模补充分布，但稀疏 pose/pseudo 置信度需加权 | 20% |
| Argoverse2 | metric ego-motion 与户外/自驾几何 | 10% |

Kinetics 只在 VGGT confidence 分布通过 RE10K 对照后，作为弱 3D pseudo label
扩展；第一版不建议混入 Stage Two 主训练。

#### 3D label 组织

每个 Stage Two sample 附加：

```text
camera intrinsics
relative pose between selected frames
depth or point map pseudo label
confidence mask
scale type: metric / normalized / unknown
```

3D 监督不作为 condition，只作为 loss target。

#### Probe 层

第一轮 probe：

```text
25% layer
50% layer
75% layer
final layer
Register-after-DiT
```

选层依据：

```text
3D probe loss
generation loss 是否退化
Stage Three frozen-probe policy warmup 效果
```

#### Loss

Stage Two loss：

```text
L_stage2 =
  L_stage1
+ λ_3d L_3d
```

其中：

```text
L_3d =
  λ_depth L_depth
+ λ_pose  L_relative_pose
+ λ_point L_point_or_correspondence
+ λ_conf  L_confidence_weighting
```

建议第一版：

```text
λ_v = 1.0
λ_fmt = 0.0
λ_3d = 0.2 initially

L_depth:
  scale-invariant log depth / masked L1

L_relative_pose:
  translation direction + rotation geodesic

L_point_or_correspondence:
  只在 pseudo label 置信度足够时启用
```

如果 3D loss 压坏 generation loss，先降低 `λ_3d` 到 `0.05`，不要马上删掉 3D
branch。

### Stage Three：VLN policy training

#### 目标

Stage Three 把 Stage Two 选出的 shared backbone/Register 表征用于导航：

```text
obs/history/Register/instruction
  -> action/nav tokens
  -> action velocity/noise
  -> denoise/decode
  -> control
```

推理时删除 generation branch：

```text
T_future = 0
```

训练时不应完全丢弃 Stage One/Two 的数据和 loss。Stage Three 是 policy 主训练，
但需要 replay/cotrain video generation 与 3D supervision，避免 shared
backbone/Register 在 VLN imitation 中遗忘长历史视频记忆和空间表征。推理仍然
policy-only，generation/3D heads 不执行。

#### 使用数据

Policy 主数据：

| 数据 | 用途 | 备注 |
| --- | --- | --- |
| R2R-CE | 标准 VLN-CE 主任务 | train/val_seen/val_unseen 结构清楚 |
| RxR-CE | 多语言/更长路径/更大规模 | guide/follower 可分阶段引入 |
| ScaleVLN | 大规模增强，适合 policy imitation | 注意与 R2R 分布关系 |
| LHPR-VLN | 长程/任务型导航补充 | batch/场景准备完成后逐步引入 |
| MP3D | R2R/RxR 共用场景 | 作为渲染/闭环环境，不是独立 instruction 数据 |

Replay/cotrain 数据：

| 数据 | replay 用途 | 备注 |
| --- | --- | --- |
| Stage One window pool | `L_visual_flow_replay`，保持 video consequence generation 与 Register memory | DL3DV、SpatialVID 20%、RE10K、Argoverse2 |
| Stage Two 3D pool | `L_3D_replay`，保持 spatial hidden/Register 表征 | DL3DV、RE10K 主导，SpatialVID/Argoverse2 低权重 |

#### VLN 样本组织

每个 sample：

```text
instruction
current observation obs_s
history prefix obs/actions up to s
target action chunk a_s...a_{s+H_nav-1}, H_nav=10
future obs: 不构建
pose/progress/distance-to-goal: 不作为 policy 输入
```

默认：

```text
H_nav = 10
loss 监督 10 个 action
closed-loop evaluation 先执行第 1 个 action
```

短轨迹不足 10 步：

```text
用 STOP padding
mask 掉 episode 结束后的无效 step
STOP loss 单独加权
```

#### Loss

Stage Three 主 loss：

```text
L_stage3 =
  λ_nav L_action_flow(A_noise -> A_out, gt_actions)
+ λ_stop L_stop_balance
+ λ_ce   L_discrete_ce_aux
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d   L_3D_replay
```

第一版建议从保守 replay 开始，具体配比需要实验：

```text
λ_nav = 1.0
λ_stop = 0.2
λ_ce = 0.1
λ_replay_video = 0.05
λ_replay_3d = 0.05

batch/task mix initial:
  VLN policy batch       70%
  Stage One video replay 20%
  Stage Two 3D replay    10%
```

待实验的配比：

| 配置 | VLN | Video replay | 3D replay | 用途 |
| --- | ---: | ---: | ---: | --- |
| S3-R0 | 100% | 0% | 0% | policy-only baseline，检查遗忘/退化 |
| S3-R1 | 80% | 15% | 5% | 轻 replay |
| S3-R2 | 70% | 20% | 10% | 默认候选 |
| S3-R3 | 60% | 25% | 15% | 强 replay，检查是否拖慢 policy 收敛 |

Replay batch 的关键约束：

```text
video/3D replay 只更新 shared backbone/Register/generation/3D heads；
policy batch 更新 shared backbone/Register/action head；
policy tokens 不能读取 replay future visual/3D；
replay loss 不把视频 pseudo action 当 policy supervision。
```

关键 mask：

```text
action/nav tokens:
  can read Register / obs / instruction / state
  cannot read future visual / future 3D
  cannot receive clean current/target action as condition

policy Register:
  cannot read future visual / future 3D

generation branch:
  can read Register / obs / action / instruction
```

#### 数据配比

Stage Three 第一版推荐：

```text
R2R-CE     25%
RxR-CE     35%
ScaleVLN   30%
LHPR-VLN   10%
```

如果 LHPR batch 或 HM3D 仍未准备完全：

```text
R2R-CE     30%
RxR-CE     40%
ScaleVLN   30%
LHPR-VLN    0%
```

EnvDrop / follower / augmentation 不应一开始全量压入。建议作为 20% 以内的
augmentation pool：

```text
标准 CE episode: 80%
augmentation:    20%
```

### 三阶段训练顺序

建议执行顺序：

#### Stage One-A：V1 smoke

```text
数据:
  DL3DV only

目标:
  跑通 [Register][local][Z_future] packing
  跑通 T_future=1 VAE encode/decode
  跑通 action interface placeholder，但不训练 policy/action loss

步数:
  1k
```

#### Stage One-B：多数据 video generation memory pretrain

```text
数据:
  DL3DV / SpatialVID / RE10K / Argoverse2

配比:
  按 window-level + history bucket 统计后设 dataset cap

目标:
  稳定 generation branch + Register long-history update
  保留 action condition / action output token 格式，但不把视频 pseudo action 训成 policy
```

#### Stage Two-A：3D probe scan

```text
数据:
  DL3DV / RE10K 主导

目标:
  冻结或半冻结 backbone，扫 probe 层
```

#### Stage Two-B：3D joint finetune

```text
数据:
  DL3DV / RE10K / SpatialVID / Argoverse2

目标:
  在 Stage One video generation loss 上直接加 3D loss；
  action interface 继续保持格式兼容。
```

#### Stage Three-A：policy-only imitation warmup

```text
数据:
  R2R-CE / RxR-CE / ScaleVLN

目标:
  generation branch 不执行；
  正式训练 action/nav output，让 action flow decoder 与离散 action decode 先收敛。

用途:
  作为 S3-R0 baseline，衡量 policy-only fine-tune 的遗忘/退化程度。
```

#### Stage Three-B：policy + Stage One/Two replay cotrain

```text
数据:
  VLN: R2R-CE / RxR-CE / ScaleVLN / LHPR-VLN
  Stage One replay: DL3DV / SpatialVID 20% / RE10K / Argoverse2 video windows
  Stage Two replay: DL3DV / RE10K / SpatialVID / Argoverse2 3D windows

默认配比:
  VLN policy batch       70%
  Stage One video replay 20%
  Stage Two 3D replay    10%

目标:
  在提升 VLN SR/SPL 的同时，保持 Register long-history video memory 和 3D spatial
  representation，避免 Stage Three 破坏 Stage One/Two 学到的 backbone 能力。
```

### 需要优先实现的文件

```text
NAV/config/v1_stage_one_video_action.yaml
NAV/config/v1_stage_two_3d.yaml
NAV/config/v1_stage_three_vln.yaml

NAV/scripts/datasets/build_v1_video_pack_manifest.py
NAV/scripts/datasets/encode_v1_vae_packs.py
NAV/scripts/datasets/build_v1_vln_policy_manifest.py

NAV/src/nav/v1/data/video_action_dataset.py
NAV/src/nav/v1/data/vln_policy_dataset.py
NAV/src/nav/v1/losses.py
```

### 当前风险

1. V1 sparse pack latent 与 V0 81-frame latent 语义不同，不能混用缓存。
2. 如果 Register/action mask 没写对，Stage Two/Three 会产生 future leakage。
3. SpatialVID 量太大，必须 weighted sampling，否则分布会压倒 DL3DV/RE10K。
4. 视频 pseudo action 与 VLN discrete action 的尺度必须通过 displacement bucket
   对齐，不能只按 frame index 对齐。
5. Stage Three 的 generation auxiliary 只能训练时使用，不能成为 policy condition。
6. 不得把 GT pose 设计成推理输入；无 pose 数据只能走弱监督或 pseudo motion
   低置信路径。


## V1 Stage One T4-IW 对齐训练规范


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-007` |
| 类型 | 训练规范（Training Specification） |
| 状态 | T4/IW Data Rule Active；Old Wan Training Entry Superseded |
| 更新时间 | 2026-08-14 |
| 职责 | 固定 V1 Stage One 中 `T_latent=4` 的数据切片、history 构造和 InfiniteWorld 对齐口径 |

### 当前结论

V1 Stage One 的新版默认 generation horizon 采用 `T_latent=4`，即每次只对一个短
future latent segment 做 RFlow / diffusion loss，而不是恢复 InfiniteWorld 的
81-frame dense target。这样保留“noisy target token 小、训练/推理更快”的核心收益。

**2026-08-14 更新**：本文中的 T4 micro chunk、IW-1/4/8/16 history 换算、
latent manifest 与 window 构造规则仍是当前有效规则；但 2026-08-13 版本的
`A_query` / direct CE action token 实现已被 DEC-034 覆盖。新的正式 action
interface 使用 `A_noise -> action flow decoder -> A_out` 和
`H_action=H_nav=10`；但 DEC-041 已进一步确认主结构必须回到
**single shared WanBlock token stream**，不采用 dual-stream / MoT-style
backbone。旧 Wan 训练入口已从当前代码删除；当前可执行结构需要按 DEC-041
重构后重新验证。

与 InfiniteWorld 的对齐只发生在 history/context span 上：

```text
InfiniteWorld chunk:
  1 IW chunk rollout = 81 RGB frames
  N IW chunks rollout = 1 + 80N RGB frames

NAV T4 micro chunk:
  1 micro chunk = 13 RGB frames -> Wan VAE latent T=4
  adjacent micro chunks stride = 12 RGB frames
  因此每个 micro step 新增 12 frames，首尾共享 1 frame
```

换算关系：

| IW-equivalent history | RGB span | T4 micro history steps |
| --- | ---: | ---: |
| 1 chunk | 81 frames | 7 |
| 4 chunks | 321 frames | 27 |
| 8 chunks | 641 frames | 54 |
| 16 chunks | 1281 frames | 107 |

因此“对齐 InfiniteWorld 16 chunks”在 V1 中不是 `history=16`，而是
`history_micro_steps=107`，target 仍然是 `T_latent=4`。

### 数据整备

正式 manifest：

```text
/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl
```

生成脚本：

```bash
python NAV/scripts/datasets/build_v1_t4_micro_manifest.py
```

当前统计口径为 `micro_frames=13`、`micro_stride=12`、`latent_t=4`。2026-08-10
已生成 manifest，统计如下：

| 项 | 数量 |
| --- | ---: |
| 总 episode | 24,261 |
| RE10K | 269 |
| DL3DV | 141 |
| SpatialVID | 23,837 |
| Argoverse2 | 14 |
| 可支撑 IW-1 history | 24,261 |
| 可支撑 IW-4 history | 15,726 |
| 可支撑 IW-8 history | 8,031 |
| 可支撑 IW-16 history | 140 |

T4 latent 编码入口：

```bash
python NAV/scripts/datasets/encode_v1_t4_micro_latents.py \
  --manifest /sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl \
  --output /sharedata/NAV/derived/v1/t4_micro_latents \
  --device cuda:0
```

每条 episode 的 payload：

```python
{
    "micro_latents": Tensor[1, N, 16, 4, 56, 112],
    "move": LongTensor[num_source_frames],
    "view": LongTensor[num_source_frames],
    "micro_frames": 13,
    "micro_stride": 12,
    "latent_t": 4,
    "num_micro_chunks": N,
}
```

smoke 已验证：

```text
/sharedata/NAV/derived/v1/t4_micro_latents_smoke/re10k/re10k__0000cc6d8b108390_000000.pt
micro_latents = [1, 8, 16, 4, 56, 112]
```

### 训练样本构造

训练时按 IW-equivalent history 采样，然后自动换算为 micro history：

```text
iw_history_chunks ∈ {1,4,8,16}
micro_history_steps = ceil((1 + 80*iw_history_chunks - 13) / 12) + 1
```

一次样本：

```text
micro chunks:
  C0, C1, ..., Ck, Ctarget

Register:
  R_{-1} = fixed R_null
  R0 = RegisterCell(R_null, concat([visual_tokens(C0), A_hist0]))
  R1 = RegisterCell(R0, concat([visual_tokens(C1), A_hist1]))
  ...
  Rk = RegisterCell(R{k-1}, concat([visual_tokens(Ck), A_histk]))
  # A_hist 以 action tokens / cross-attention 方式参与；禁止 action/latent bias

local_memory:
  latest latent cell of Ck -> [B,16,1,H,W]

generation target:
  Ctarget -> [B,16,4,H,W]

loss:
  RFlow / diffusion loss only on Ctarget
```

Action 接口暂保持 InfiniteWorld 兼容：target micro window 的 13 帧 `move/view`
pad 到 81，并传给原 `ActionEncoder`。这是兼容旧 backbone 的工程选择；后续 V1
action-stream 结构实现后再替换。

2026-08-11 正式 Stage One 已升级为 V1 action interface：

```text
A_hist:
  history window 的 move/view/action 编成独立 A_hist tokens；
  进入 RegisterCell，帮助解释历史视觉变化；
  禁止 action bias / latent bias 注入。

Register:
  R_{-1} = fixed R_null
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  RegisterCell 读取视觉 latent tokens 与 A_hist tokens；
  不读取 pose / odometry 原值；
  不把 action embedding 加到 latent/Register/patch tokens 上作为 bias。

A_cur:
  target micro window 的 move/view 作为 video generation current action
  condition；正式 V1 结构中构造成 DiT condition/context token，追加到 Wan
  cross-attention context。原 InfiniteWorld `action_encoder` 在正式 V1 中接收
  no-op，避免 action 同时以 latent bias 和 condition 双路注入。

A_noise / A_out:
  `A_noise` tokens 进入 single shared WanBlock token stream；
  `A_out` 从 shared WanBlock 的 action-token hidden 经 Giga/DreamZero-style
  action flow decoder 读出。
  Stage One 中 `lambda_action=0`，不计算 policy/action loss。
```

因此 Stage One 只训练 video generation memory，不把 video pseudo action 训练成
policy。真正的 action output supervision 在 Stage Three 用 VLN 数据训练。
旧版 `Register/local -> 旁路小 action head` 是错误 scaffold，已废弃，不能作为
正式 Stage One 或 policy latency 结论。

### Text-conditioned fullmix 5000-step 正式训练（2026-08-12）

本轮按“iWorld-Bench 对齐”的完整数据配比启动 V1 Stage One：

```text
run_name =
  v1-stageone-text-fullmix-wan21official-lp-mb1-ebs16-5000-20260812-020056

tmux =
  nav_v1_stageone_text_full5000

pipeline log =
  NAV/log/v1-stageone-text-fullmix-wan21official-lp-mb1-ebs16-5000-20260812-020056.pipeline.log

TensorBoard =
  0.0.0.0:6013
```

启动逻辑：

```text
1. GPU1 缓存 SpatialVID text condition
   source:
     /sharedata/datasets/SpatialVID/raw/huggingface/SpatialVID/annotations/*/*/{caption.json,instructions.json}
   output:
     /sharedata/NAV/derived/v1/text_embeddings/t4_micro/spatialvid/*.pt

2. text cache 完成后自动启动 5000 optimizer steps 训练
   script:
     NAV/scripts/run_v1_stageone_text_full5000.sh
```

Text 规则：

```text
SpatialVID:
  使用 SceneSummary / SceneDescription / CameraMotion / ShotImmersion
  + instructions.json 中的 camera instruction，经过 UMT5 编码后作为 DiT text
  condition。

DL3DV / RE10K / Argoverse2:
  当前没有统一 caption，fallback 到 empty UMT5 condition。
```

历史 Wan 训练脚本曾支持 per-sample text sidecar：

```text
旧 Wan T4 训练入口（2026-08-14 已从当前代码删除）
  --text-cache-root /sharedata/NAV/derived/v1/text_embeddings/t4_micro
```

缺失 text sidecar 时会自动 fallback 到
`/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt`，并在 train log 与
TensorBoard 记录 `text/fallback_rate`。这保证新增 text 不会破坏无文本数据集的
训练。

正式训练额外设置：

```text
--require-text-datasets spatialvid
```

因此 SpatialVID 只有在存在 text sidecar 时才会进入训练；DL3DV / RE10K /
Argoverse2 因当前无统一 caption，允许使用 empty UMT5 fallback。这个规则避免
GPU0 后台新落盘但尚未缓存 text 的 SpatialVID 样本混入本轮正式训练。

完整数据配比：

| 数据集 | 目标权重 |
| --- | ---: |
| SpatialVID | 45% |
| DL3DV | 35% |
| RE10K | 15% |
| Argoverse2 | 5% |

当前可用 latent 根目录为：

```text
/sharedata/NAV/derived/v1/t4_micro_latents_spatial20
```

启动时已确认该目录含：

```text
DL3DV:      141 episodes
RE10K:      269 episodes
SpatialVID: 3869 episodes with text sidecar
Argoverse2: 0 episodes
```

因此配置中保留 Argoverse2=5%，但实际 sampler 会在当前可用 history bucket 内
自动重归一化；等 Argoverse2 T4 latent 落盘后，下一轮重新扫描即可进入训练。

训练启动审计：

```text
candidate_files = 4319
used_files = 4279
skipped_spatialvid_missing_required_text = 40
num_training_windows = 388,994
step_1_loss = 1.3859
step_1_history = IW-16 / 107 micro updates
step_1_peak_reserved = 32.78 GiB
```

训练超参：

```text
initialization:
  official Wan2.1-T2V-1.3B safetensors

variant:
  latent_prefix

train_scope:
  full parameter training

history_iw_chunks:
  1,4,8,16

batch:
  micro batch size = 1
  gradient accumulation = 16
  effective batch size = 16

optimizer:
  AdamW
  lr = 1e-5
  weight_decay = 0.01
  grad_clip = 1.0
  dtype = bf16

save:
  every 1000 optimizer steps
  final checkpoint = full-final.pt
```

### 训练入口状态

2026-08-14 之后，本文不再定义当前可执行训练入口；本文只保留 T4/IW 对齐和
历史 run 配置。当前完整模型链路验证入口为：

```bash
bash NAV/scripts/run_v1_full_pipeline_smoke.sh cpu
```

旧 Wan T4 入口已从当前代码删除。旧默认配置仅作为历史记录：

```text
checkpoint = /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
variant = latent_prefix
train_scope = full
target_latent_t = 4
batch_size = 1
gradient_accumulation_steps = 16
history_iw_chunks = 1,4,8,16
```

smoke 记录：

```text
run_dir = NAV/log/v1-wan-stage1-t4-smoke-20260810-152404
history_iw_chunks = 1
history_micro_steps = 7
target_latent_t = 4
loss = 0.1317225
peak_reserved_gib = 30.32 GiB
```

正式 run：

```text
run_dir =
  NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650

tmux =
  nav_v1_stageone_final_train

TensorBoard =
  0.0.0.0:6012

initialization =
  official Wan2.1-T2V-1.3B safetensors
  loaded_keys=825
  patch_embedding.weight:
    Wan native 16 latent channels -> 1536 hidden dim
    shape-match direct load; no 20-channel mask concat
  token_type_embedding:
    newly initialized; marks prefix / future / action token roles
```

### 当前限制

- 当前正式 Stage One 训练脚本采用 GigaWorld-Policy-style per-token timestep：
  clean prefix / Register / local visual tokens 为 `t=0`，future visual 使用
  RFlow `visual_t`，`A_noise` action tokens 使用独立 `action_t`。现有代码仍需
  从旧 `A_query` smoke / dual-stream 残留迁移到 DEC-041 的 shared WanBlock +
  `A_noise` action flow decoder。
- 正式 long-history 已覆盖 IW-16：107 个 T4 micro history updates，target 是
  第 108 个 micro chunk，用于对齐 InfiniteWorld 的 16-chunk teacher history。
- Stage One action output branch 只保留格式，不训练 policy/action loss；
  Stage Three 才训练 action output。
- 当前 T4 route 是 Stage One generation branch；Stage Two 3D probe 和 Stage
  Three policy-only 裁剪尚未在该脚本中实现。

### Giga timestep 历史 smoke（2026-08-13）

该 smoke 已被 DEC-034 设定覆盖，只保留为历史实现记录。

```text
run_dir =
  NAV/log/v1-final-giga-timestep-smoke-20260813

checkpoint =
  full-step-000001.pt

model =
  official Wan2.1-T2V-1.3B initialization
  16-channel patch embedding
  token_type_embedding(prefix / future / action)
  A_cur noisy-future-only condition
  A_query main-stream action tokens with independent action_t
  # historical smoke only; formal DEC-041 uses A_noise + shared WanBlock action decoder

loss =
  L_visual_flow only
  lambda_action = 0

smoke result =
  loss = 1.1692
  visual_timestep_mean = 917.26
  action_timestep_mean = 998.0
  peak_reserved_gib = 30.95 GiB
```

### 旧 fullmix 训练计划（2026-08-13，已被 DEC-034 覆盖）

```text
status =
  historical / diagnostic; not the final formal structure after DEC-034

run_dir =
  NAV/log/v1-final-giga-timestep-fullmix-wan21official-lp-mb1-ebs16-5000-20260813-210644

tmux =
  nav_v1-final-giga-timestep-fullmix-wan21official-lp-mb1-ebs16-5000-20260813-210644

initialization =
  /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors

train_scope =
  full parameters

data_root =
  /sharedata/NAV/derived/v1/t4_micro_latents_spatial20

dataset_weights =
  spatialvid=0.45, dl3dv=0.35, re10k=0.15, argoverse2=0.05

history_iw_chunks =
  1,4,8,16

timestep_schema =
  clean prefix/Register/local: t=0
  future visual: visual_t, RFlow shift=7.0
  A_query/action: action_t, Giga-style shift=5.0
  # superseded by A_noise/action flow decoder in DEC-034

batch =
  physical batch size = 1
  gradient accumulation = 16
  effective batch size = 16

steps =
  5000 optimizer steps

save_every =
  1000 optimizer steps

first logged step =
  step = 1
  loss = 1.2101
  history_iw_chunks = 16
  history_micro_steps = 107
  visual_timestep_mean = 814.65
  action_timestep_mean = 683.0
  peak_reserved_gib = 33.11 GiB
```

### 本轮结束后的测评与续训顺序

当前 1000 optimizer steps run 结束后，先固定 checkpoint 做 generation quality
check，再进入目标数据集混合训练。

最小测评要求：

```text
checkpoint:
  full-final.pt 或 full-step-001000.pt

history buckets:
  IW-1 / IW-4 / IW-8 / IW-16

检查项:
  视频文件可解码
  非黑屏
  有像素变化
  与 history/local observation 连贯
  不同 history 长度下 loss 与生成样例分开记录
```

通过基本生成质检后，下一轮 mixed target dataset training 才重新扫描
`/sharedata/NAV/derived/v1/t4_micro_latents_spatial20/`，纳入当前后台继续落盘的
SpatialVID 20% latent。下一轮训练必须记录 dataset-aware sampler 或数据配比；
不得按自然文件数让 SpatialVID 压倒 DL3DV/RE10K。

历史自动评估入口（已从当前代码删除）：

```text
script:
  旧 V1 T4 generation sanity 脚本（已删除，历史见 git）

watcher tmux:
  nav_v1_stageone_final_eval_waiter

output:
  NAV/result/stageone_t4_generation_sanity/
    v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/

log:
  NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650.eval_waiter.log
```

该 watcher 会等待 `full-final.pt` 出现，再用 GPU1 对 IW-1/4/8/16 各抽 1 个样例
生成 `gen.mp4` / `gt.mp4` 和 `metrics.json`。


## V1 Stage Two RE10K Pose-only 正式验证训练


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| ID | NAV-TRN-009 |
| 类型 | 训练实验记录 / Stage Two formal verification |
| 状态 | Blocked / Superseded by DEC-041 |
| 更新时间 | 2026-08-14 |
| 职责 | 记录只使用 RE10K、但模型结构/latent 几何/loss 形式均按 V1 最终标准执行的 Stage Two pose-only 验证训练 |

### 当前结论

本轮原计划是 Stage Two 正式验证训练，不是小型化 diagnostic。除数据集限制为
RE10K 外，其余约束保持当时 V1 标准。但 2026-08-14 下午复查发现，当前代码中的
`V1DualStreamBackbone` 违反 DEC-041：主结构不应采用 GigaWorld-style dual-stream /
MoT-style backbone，而应是 single shared WanBlock token stream。因此本 run
不能继续作为正式 Stage2 主线，只保留为发现结构错误和显存问题的审计记录。

```text
tmux:
  nav_stage2_re10k_poseonly_formal  # 已停止

run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_*_20260814_*/

script:
  NAV/scripts/run_v1_stage2_re10k_pose_video.sh

entrypoint:
  NAV/scripts/train_v1_stage2_re10k_pose_video.py
```

已完成启动前检查：

```text
preflight:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/formal_preflight.json

Wan init audit:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/wan_init_audit.json
```

### 正式约束

本轮不允许以下小型化：

```text
不 resize latent
不缩小 latent/video 空间分辨率
不缩小 backbone depth
不缩小 hidden width
不缩小 Register token 数
不启用 depth/point pseudo target
不使用旧 diagnostic checkpoint
```

允许缩小的只有数据范围：本轮只用 RE10K。

### 模型结构

```text
base init:
  /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors

hidden_dim:
  1536

num_backbone_layers:
  30

num_heads:
  12

mlp_ratio:
  8960 / 1536

Register:
  128 tokens
  R_null fixed template
  RegisterCell recurrent update

Action interface:
  H_action = 10
  A_hist participates in Register update as tokens
  A_cur is video generation condition
  A_noise action tokens exist but action loss = 0 in Stage Two

Stage Two 3D:
  current hidden -> PoseHead
  depth supervision disabled
```

Wan 初始化审计：

```text
loaded keys:
  242

skipped:
  0

loaded scope in this superseded implementation:
  patch_embedding.weight/bias
  Wan blocks.0-29 self_attn qkv/o -> V1 visual_self
  Wan blocks.0-29 ffn -> V1 visual_ffn
```

说明：上述加载方式正是被 DEC-041 废弃的实现证据之一。它只把 Wan 权重加载到
形状兼容的 visual stream，而不是复用同一套 shared WanBlock；因此 loaded keys
显著少于此前 Stage One Wan/latent-prefix 路径。下一版 Stage2 formal 必须回到
single shared WanBlock token stream，再重新做 Wan init audit。

### 失败原因 / Why this run stopped

这次不是 pose loss 本身导致显存问题。`PoseHead` 只有百万级参数，真正的问题是
当前 `V1DualStreamBackbone` 将 action path 做成了额外 transformer/expert，
导致模型参数约 `2.89B`，backbone 约 `2.79B`；而此前可训练 Stage One 路径约
`1.42B`。AdamW 在 step 1 后创建 optimizer state，step 2 backward 稳定 OOM。

因此结论是：

```text
错误原因:
  dual-stream / MoT-style backbone 残留

不是原因:
  RE10K pose target
  current hidden -> PoseHead
  T4 latent geometry

下一步:
  按 DEC-041 改为 single shared WanBlock token stream
  保留最新 Stage2 数据、pose-only loss、current-chunk supervision 和 action-token 接口
```

### DEC-041 后代码修正状态

已完成静态代码修正：

```text
model:
  NAV/src/nav/v1/models/full_model.py

backbone:
  V1DualStreamBackbone -> V1SharedWanBackbone

shared token order:
  [Register][Z_obs][Z_future^σ][A_noise][text][A_cur]

mask:
  action/Register/obs 不读 future
  A_cur 只允许被 future visual tokens 读取

parameter audit:
  total ≈ 1.217B
  backbone ≈ 1.109B

syntax:
  py_compile passed
```

注意：这只是代码结构修正和静态检查，不等同于正式 Stage2 训练完成。下一次正式
Stage2 RE10K pose-only 训练必须使用 shared WanBlock 版本重新启动，并重新生成
Wan init audit、TensorBoard、checkpoint 和 eval 记录。

### Shared WanBlock 2k 正式测试 run（2026-08-14）

已按 DEC-041 后的新结构启动 2000 optimizer steps Stage2 测试：

```text
tmux:
  nav_stage2_re10k_poseonly_formal

run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648/

tensorboard:
  port 6016

command:
  STEPS=2000
  BATCH_SIZE=1
  GRAD_ACCUM=16
  HISTORY_STEPS=1
  MAX_TRAIN_SAMPLES=0
  NUM_WORKERS=0
  SAVE_EVERY=200
  EVAL_EVERY=100
  LOG_EVERY=10
  DEVICE=cuda:0
  bash scripts/run_v1_stage2_re10k_pose_video.sh v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648
```

启动前检查：

```text
backbone_class:
  V1SharedWanBackbone

total parameters:
  1,216,983,212

dataset:
  RE10K only, 269 samples

latent:
  [16,4,56,112]

pose target:
  [4,9]

effective batch:
  16
```

首个 optimizer step 已写出：

```text
step = 1
loss = 2.048828125
loss_visual = 2.00439453125
loss_pose = 0.4259033203125
cuda_max_memory_gb = 35.44545888900757
```

step10 已写出，确认 shared WanBlock + AdamW 已越过旧 dual-stream 版本 step2 OOM：

```text
step = 10
seconds_per_step_window = 74.24005193710327
loss = 1.994140625
loss_visual = 1.95703125
loss_pose = 0.3751220703125
cuda_max_memory_gb = 39.97349309921265
```

当前状态：进程仍在运行，GPU0 约 43.6GiB / 49.1GiB、99–100% utilization。
按 step10 窗口速度粗估，2000 optimizer steps 约需 41–43 小时，另加 eval 与
checkpoint 开销。

### 数据与样本构造

```text
dataset:
  RE10K only

latent root:
  /sharedata/NAV/derived/v1/t4_micro_latents/re10k

manifest:
  /sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl

camera annotation:
  /sharedata/RealEstate10K/src_annotations/train/cameras

matched samples:
  269
```

每个样本使用完整 T4 micro latent：

```text
micro_latents:
  [1, 13, 16, 4, 56, 112]

history_latents:
  [S, 16, 4, 56, 112]

z_obs:
  [16, 4, 56, 112]

z_future_noisy / z_future_noise / z_future_target:
  [16, 4, 56, 112]

pose_target:
  [4, 9]
  [tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]
```

样本边界：

```text
Register history = C_0 ... C_{t-2}
Current / Local  = C_{t-1}
Target future    = C_t
```

当前启动参数 `history_steps=1`，所以窗口为：

```text
C_i -> Register update
C_{i+1} -> current/local + pose supervision
C_{i+2} -> future video generation target
```

Pose 来自 RE10K 落盘 camera pose / intrinsics，不作为模型输入，只作为监督：

```text
/sharedata/RealEstate10K/src_annotations/train/cameras/<scene_id>.txt
```

### Loss

本轮是 pose-only Stage Two：

```text
L_stage2 =
  L_visual_flow
+ λ_pose L_pose

λ_pose = 0.1
λ_depth = 0.0
```

其中：

```text
L_visual_flow:
  MSE(future_velocity, z_future_target - z_future_noisy)

L_pose:
  masked MSE(PoseHead(current hidden), RE10K relative pose)
```

2026-08-17 更新：本节旧版实现曾错误地把 `z_future_noisy` 直接设为纯
Gaussian noise，并监督：

```text
MSE(future_velocity, z_future_target - z_future_noisy)
```

这与 InfiniteWorld/Wan 的 RFlow 训练/采样不一致，导致 step2000 评测中 video branch
几乎只输出彩色噪声。现已按 DEC-043 修正为 InfiniteWorld-style reversed velocity：

```text
t_raw ~ Uniform(0,1)
t     = shift * t_raw / (1 + (shift - 1) * t_raw), shift=7.0
noise = N(0,I)
x_t   = (1 - t) * z_future_target + t * noise

model input:
  z_future_noisy = x_t
  visual_timestep = t

L_visual_flow:
  MSE(future_velocity, noise - z_future_target)

one-step x0 estimate for eval:
  z_pred = x_t - t * future_velocity

multi-step sampler:
  z_1 ~ N(0,I)
  z_{k+1} = z_k - future_velocity(z_k, t_k) * (t_k - t_{k+1})
```

该修正只改变 video branch 的 noisy latent / velocity target / sampler；不改变：

- `RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`；
- `A_hist` / `A_cur` / `A_noise` token 结构；
- single shared WanBlock token stream 与 causal mask；
- current hidden pose supervision；
- action flow decoder / policy token 位置。

完整模型 1-step formal 链路检查已通过：

```text
run:
  NAV/log/v1_stage2_re10k_pose_video/debug_rflow_video_branch_fullmodel_1step_20260817_alloc/

result:
  full model + full RE10K T4 latent + Wan init + forward/backward/update ok
  video_objective = rflow_reversed_velocity
  rflow_shift = 7.0
```

不声明 depth/point 表现。

### 训练超参

```text
steps:
  1000 optimizer steps

batch:
  micro_batch = 1
  gradient_accumulation = 16
  effective_batch = 16

optimizer:
  AdamW
  lr = 1e-5
  weight_decay = 0.01
  grad_clip = 1.0

dtype:
  bf16

activation checkpointing:
  enabled

save:
  every 200 optimizer steps

eval:
  every 100 optimizer steps
```

正式结构 GPU preflight：

```text
run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_gpu_preflight_ckpt_20260814_092313/

configuration:
  full model + full latent + batch=1 + grad_accum=1 + bf16

result:
  peak cuda memory = 39.73 GiB
  step time = 5.03 s
  Wan loaded keys = 242
```

正式训练 step 1：

```text
step:
  1 / 1000

seconds_per_optimizer_step:
  75.43 s

peak cuda memory:
  39.93 GiB

loss:
  train/loss = 1.9746
  train/loss_visual = 1.9297
  train/loss_pose = 0.4614
```

### 查看命令

```bash
tmux attach -t nav_stage2_re10k_poseonly_formal

tail -f /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/train.jsonl

tensorboard --logdir /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/tensorboard --host 0.0.0.0 --port 6016
```

### 验收口径

本轮只看 pose-only Stage Two 是否有真实效果：

```text
1. train/eval loss_visual 是否下降；
2. train/eval loss_pose 是否下降；
3. final checkpoint 上 pose_translation_mae / pose_rotation_angle_deg / pose_fov_mae；
4. generation branch 后续用完整 latent + VAE decode 做 sanity，不用降采样结果验收；
5. depth/point 不在本轮验收范围。
```

### 原版 cotrain step2000 正式评测（2026-08-16）

对应 run：

```text
log:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648/

checkpoint:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648/checkpoints/step_002000.pt

result:
  NAV/result/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_step2000_detailed_eval/
```

评测使用完整 `V1FullWorldNavModel`、完整 RE10K T4 latent 几何与 RE10K camera
pose 标注；没有降模型、降分辨率或 scaffold。RGB 评测为 one-step denoise
reconstruction proxy：

```text
z_pred = z_future_noisy + future_velocity
```

因此它衡量当前训练目标是否能把 noisy future latent 拉回 GT future latent，不等价于
完整 multi-step diffusion sampling。

全量 RE10K eval 269 条得到：

| 指标 | 数值 |
| --- | ---: |
| `loss_visual_velocity_mse` | 1.55817 |
| `latent_future_mse` | 1.55817 |
| `baseline_noise_future_mse` | 1.61711 |
| `latent_mse_improvement_vs_noise` | 0.05894 |
| `future_velocity_rms / target_velocity_rms` | 0.31497 / 1.27032 |
| `latent_future_cosine` | 0.09206 |
| `pose_translation_mae` | 0.04552 |
| `pose_rotation_angle_deg` | 9.31405° |
| `pose_fov_mae` | 0.07647 rad |

RGB decode 使用确定性前 64 条，保存前 8 组 `obs / gt_future /
pred_future_proxy` 视频：

| 指标 | 数值 |
| --- | ---: |
| `rgb_mse` | 0.09517 |
| `rgb_l1` | 0.24941 |
| `rgb_psnr_db` | 10.39 dB |
| `rgb_global_ssim` | 0.01364 |
| `rgb_temporal_absdiff_pred / target` | 0.11944 / 0.02594 |

结论：

- 3D pose supervision 训动了；当前 3D 监督从 `obs_hidden` 读出，也就是 current
  chunk hidden token，不是 Register/future/action token。
- video branch 基本没有形成有效 denoise/generation：相对纯 noise baseline 的
  latent MSE 改善只有约 0.059，预测 velocity RMS 只有 target velocity RMS 的约
  24.8%。
- 解码视频不是黑屏，但 `pred_future_proxy` 基本是彩色噪声；contact sheet 与
  temporal diff 均支持该判断。
- 因此该 run 的正式结论是：**pose probe/head 可以从 current hidden 中读出
  frame-level pose，但当前 video diffusion formulation 未学会有效生成。**

后续必须优先检查 video training formulation：`z_future_noisy` 是否应按 timestep
构造为 `x_t=(1-t)z_target+t noise`，`future_velocity` target 是否与 RFlow/Wan
scheduler 一致，以及当前 one-step proxy 与最终 multi-step sampling 是否训练-推理一致。

该问题已在 2026-08-17 按 DEC-043 修正。上表数值仍保留为“错误 video formulation
下的失败基线”，不能再作为新版 RFlow video branch 的质量结论。


## V1 Stage Two RE10K 视频生成与 3D 监督 Pilot


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| ID | NAV-TRN-008 |
| 类型 | 训练实验记录 / Stage Two pilot |
| 状态 | Diagnostic invalid for acceptance / Superseded |
| 更新时间 | 2026-08-14 |
| 职责 | 记录使用 RE10K 与 Wan2.1 official init 直接测试 Stage Two 视频生成 + 3D 监督是否可联合训练的入口、口径、限制和验收方式 |

### 当前结论

本轮已经完成一次 Stage Two diagnostic run，但由于它缩小了 latent 空间分辨率并
缩小了 backbone depth，**不得作为 Stage Two 完整训练、正式 smoke、验收结果或
方案有效性证据**。它只保留为调试记录，说明代码路径曾经可以运行；后续所有
Stage Two 训练必须使用完整模型结构和完整 latent/video 几何。

对应记录：

```text
run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/

script:
  NAV/scripts/run_v1_stage2_re10k_pose_video.sh

entrypoint:
  NAV/scripts/train_v1_stage2_re10k_pose_video.py
```

该 run 使用当前 `V1FullWorldNavModel`、RE10K T4 micro latent、RE10K camera
annotation、正式 forward/backward、TensorBoard、eval 和 checkpoint 逻辑；但为降低
成本缩小了 latent 空间分辨率：

```text
原始 T4 latent:
  [C=16, T=4, H=56, W=112]

pilot 输入:
  [C=16, T=4, H=8, W=16]
```

同时 `num_backbone_layers=2`，不等于正式 Wan/V1 backbone 深度。因此本轮不得再
表述为“完整链路 pilot 合格”；所有指标只能作为 invalid diagnostic 附录。

### 训练配置

```text
initial weight:
  /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors

data:
  latent root:
    /sharedata/NAV/derived/v1/t4_micro_latents/re10k
  manifest:
    /sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl
  camera annotation:
    /sharedata/RealEstate10K/src_annotations/train/cameras

matched samples:
  269 条 RE10K T4 latent episode/window

model pilot size:
  hidden_dim=1536
  num_backbone_layers=2
  num_heads=12
  patch_size=(1,2,2)
  register_tokens=16
  action_horizon=10

optimizer:
  AdamW lr=1e-5 weight_decay=0.01

batch:
  micro_batch=4
  grad_accum=4
  effective_batch=16

steps:
  1000 optimizer steps
  save_every=200
  eval_every=100
```

Wan 初始化不是完整原生 Wan DiT 复用，因为当前 V1 backbone 已经改成
dual-stream / Register / action-compatible 结构。脚本会加载形状兼容的 Wan
视觉组件，并写入审计：

```text
NAV/log/v1_stage2_re10k_pose_video/<run>/wan_init_audit.json
```

当前 `layers=2` pilot 中已确认加载：

```text
patch_embedding.weight/bias
Wan blocks.0/1 self_attn qkv/o -> V1 visual self-attn
Wan blocks.0/1 ffn -> V1 visual ffn
```

RegisterCell、ActionTokenEncoder、pose/depth heads、action decoder 等 V1 新结构随机
初始化并参与训练。

### 样本构造

本轮沿用 V1 流式边界：

```text
给定 micro chunks:
  C_0, C_1, ..., C_{t-2}, C_{t-1}, C_t

预测 C_t:
  Register history = C_0 ... C_{t-2}
  Current / Local  = C_{t-1}
  Target future    = C_t
```

当前 pilot 设置 `history_steps=1`，所以每个样本为：

```text
history_latents = C_i
z_obs           = C_{i+1}
z_future_target = C_{i+2}
z_future_noisy  = N(0, I)
```

`A_hist` 与 `A_cur` 来自 RE10K pose-derived motion bucket 中的 `move` 标签：

```text
a_hist_primitives:
  参与 RegisterCell update，不走 additive bias

a_cur_primitives:
  作为 video generation branch 的 condition token

a_noise:
  action-compatible policy token 格式占位；Stage Two pilot 不监督 policy action
```

Text 使用 empty text tokens；RE10K 本轮不提供 text instruction。

### Loss 设计

本轮 Stage Two forward 仍然是视频生成前向，只额外从 current hidden 读 3D：

```text
Register history + z_obs + z_future_noisy + A_cur/A_noise
  -> V1 historical dual-stream backbone  # superseded by DEC-041
  -> FutureLatentHead
       L_video = MSE(v_pred, z_future_target - z_future_noisy)
  -> PoseHead(current hidden)
       L_pose = masked MSE(pose_pred, pose_target)
  -> DepthHead(current hidden)
       L_depth = masked MSE(depth_pred, depth_target)

L_total = L_video + λ_pose L_pose + λ_depth L_depth
```

`pose_target` 是 current chunk 内 `T_latent=4` 个 latent frame 对应的 RE10K camera
relative pose：

```text
[tx, ty, tz, qw, qx, qy, qz, fov_x, fov_y]
```

它由 `/sharedata/RealEstate10K/src_annotations/train/cameras/<scene>.txt`
中的 camera pose / intrinsic 解析得到，并相对 current chunk 第一个 anchor frame
归一化。

`DepthHead` 已接入正式模型和 loss 接口，但本轮 `depth_mask=0`、`λ_depth=0`。
原因是 RE10K 原始数据没有 GT depth，而当前本地 VGGT pseudo-depth run 只覆盖少量
scene 的 6 张选择帧，无法可靠逐帧对齐到所有 current chunk。后续若补齐
VGGT/VGGT-Ω pseudo depth 或使用带 GT depth 的 simulator/DL3DV 派生标签，即可把
`depth_mask` 与 `λ_depth` 打开。

### 验收指标

本轮验收分两层。

第一层是链路验收，已经通过：

```text
full model smoke:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_022520/report.json

RE10K + Wan 2-step smoke:
  NAV/log/v1_stage2_re10k_pose_video/debug_2step/

RE10K + Wan batch4/layers2 smoke:
  NAV/log/v1_stage2_re10k_pose_video/debug_bs4_layers2/
```

已确认：

- `loss_visual`、`loss_pose` 可以同时 forward/backward；
- `pose_head`、`depth_head` 接入正式模型；
- checkpoint / eval / TensorBoard 写入正常；
- Wan compatible init 审计可追溯。

第二层是训练验收，等待 1000-step run 完成后检查：

```text
生成表现:
  eval/loss_visual 与 train/loss_visual 稳定下降；
  取 checkpoint 生成 z_future 并经 VAE decode，检查非黑屏、非静态、与 current
  view/运动趋势一致。

3D 表现:
  eval/loss_pose 稳定下降；
  pose translation / rotation / FoV 的分项误差低于随机初始化基线；
  depth 仅在后续补齐 depth label 后验收，本轮不把 depth=0 记为合格。
```

### Diagnostic 结果（不得作为验收证据）

训练已于 2026-08-14 02:31 CST 完成 1000 optimizer steps：

```text
final checkpoint:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/checkpoints/step_001000.pt

eval result:
  NAV/result/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/eval_metrics.json

sample tensor:
  NAV/result/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/eval_samples.pt
```

训练内置 eval 曲线：

| step | eval/loss_visual | eval/loss_pose |
| ---: | ---: | ---: |
| 100 | 0.7931 | 0.00191 |
| 200 | 0.4664 | 0.00186 |
| 500 | 0.4429 | 0.00139 |
| 700 | 0.4321 | 0.00129 |
| 900 | 0.4259 | 0.00161 |
| 1000 | 0.4264 | 0.00185 |

独立 eval 脚本在 32 个 batch 上得到：

```text
latent_future_mse:       0.44093
latent_future_l1:        0.51441
pose_translation_mae:    0.02297
pose_rotation_angle_deg: 3.43794
pose_fov_mae:            0.05505 rad
```

仅作为调试观察：

- pose loss 在小模型/低分辨率 diagnostic 中可以下降；
- latent-space video loss 在小模型/低分辨率 diagnostic 中可以下降；
- 不声明 pose supervision、视频生成或 Stage Two 方案已验收；
- depth 未参与，后续 Stage Two pose-only 不再纳入 depth。

### 运行查看

```bash
tail -f NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/train.jsonl

tensorboard --logdir NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/tensorboard --host 0.0.0.0 --port 6015
```

### 下一步

1000 steps 完成后需要做三件事：

1. 分析 `loss_visual / loss_pose / eval` 的收敛曲线；
2. 用 checkpoint 生成并 decode 一批 RE10K 样本，放入 `NAV/result/v1_stage2_re10k_pose_video/<run>/`；
3. 若 pose 合格但生成差，优先提高 latent 空间分辨率或增加 native Wan block 复用；
   若生成合格但 pose 差，优先补 VGGT/VGGT-Ω depth/point target，并做 hidden layer
   probe。


## V0 Register 流式训练样本与在线状态语义


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-003` |
| 类型 | 训练语义规范（Training Semantics Specification） |
| 状态 | Historical Semantics / V0 only |
| 更新时间 | 2026-07-29 |
| 职责 | 定义单轮、多轮、teacher forcing、Register递归和训练/推理差异 |

### 当前采用的训练样本

本节描述 V0 Register 训练语义；当前 V1 主线以 `RegisterCell` 和
full-pipeline smoke 文档为准。

当前 SpatialVID Short 使用 random-prefix next-chunk teacher forcing。一个
样本是同一 episode 中连续的2或3个 chunk：

```text
2 chunks: [history C0]     → target C1
3 chunks: [history C0,C1]  → target C2
```

离线数据只提供干净 VAE latent \(Z_i\) 和目标 action \(A_i\)。Register 在
forward 内从 checkpoint 中的 learnable initial state 开始在线递归。

以上是现有 A/B 原型的实现事实，不是最终 Stage One 设计。下一次结构改造必须
替换为：

```python
register = extractor(chunks[0])
for history in chunks[1:-1]:
    register = updater(register, history)
```

即第一个 History Chunk 负责提取初始场景 Register，后续 History Chunk 才调用
Updater；不存在可学习的场景初始 Register。

### 两 Chunk 的完整前向

\[
R_1=\operatorname{Extractor}_\theta(Z_0)
\]

local memory 为 history 最后一个 latent frame：

\[
L_0=Z_0[:,:,-1:]
\]

只对目标 \(Z_1\) 采样 diffusion timestep 和噪声：

\[
X_{1,t}=\alpha_t Z_1+\sigma_t\epsilon
\]

模型预测：

\[
\hat v_1=D_\theta(X_{1,t},t,R_1,L_0,A_1,\mathrm{text})
\]

当前原型的 `detach_history=false` 使 loss 可反传至 Register 模块；改造后该
梯度应反传至 Extractor。VAE latent 是常量，不训练 VAE。

### 三 Chunk 与更多历史

三 chunk 样本：

\[
R_1=E(Z_0),\quad R_2=U(R_1,Z_1)
\]

\[
\hat v_2=D(X_{2,t},t,R_2,L_1,A_2,\mathrm{text})
\]

训练不先执行完整 diffusion sampling 来生成 \(C_1\)。历史直接使用真实干净
latent，这就是 teacher forcing。每个样本只有一次昂贵 DiT forward/backward，
历史长度只增加较轻的 Register updates。

当前代码对任意 \(N\)-chunk窗口采用：

```python
register = extractor(chunks[0])
for history in chunks[1:-1]:
    register = updater(register, history)
local = chunks[-2][:, :, -1:]
loss = diffusion_loss(chunks[-1], register, local, target_action)
```

### A/B 注入差异

#### A / latent prefix

Register 为固定：

```text
[B,16,4,H,W]
```

每个空间位置的4个 Register planes 作为 query，读取 history chunk 经时间池化
得到的8个 token。DiT latent 侧为：

```text
[4 Register planes; 1 local plane; noisy target]
```

#### B / DiT condition

history chunk 池化为空间/时间观测 token，16个 `[256]` Register token 作为
query 更新。随后投影为4096维并与 UMT5 文本拼接：

```text
latent side:    [1 local plane; noisy target]
condition side: [text tokens; Register tokens]
```

两者都保留 Infinite-World local memory，只替换 HPMC history。

### 训练与推理的差异

训练：

```text
真实 C0 → Register
真实 C1 → Register
...
Register + 最新真实 local frame → 预测下一真实 chunk 的 diffusion target
```

推理：

```text
生成 C0 → Register
生成 C1 → Register
...
Register + 最新生成 local frame → diffusion sampling 下一 chunk
```

训练时历史真实、推理时历史由模型生成，存在 autoregressive exposure bias。
当前先用 teacher forcing 保证稳定和效率；后续再评估：

- 少比例 scheduled sampling；
- 离线 rollout latent；
- 单步 predicted-\(x_0\) 更新；
- 对生成 history stop-gradient。

不得在没有成本评估时让训练为每个 history chunk 执行完整几十步 diffusion
sampling。

### Last-target 与 Dense-target

当前采用 last-target：

```text
C0,C1,C2,C3,C4 → C5
```

只对 \(C_5\) 计算 diffusion loss，但梯度可经过全部 Register updates。优点是
一次样本只运行一次 DiT。

未来可选 dense-target：

```text
C0 → C1
C0,C1 → C2
...
C0,...,C4 → C5
```

它仍可使用真实 history，不需要生成历史，但一个6-chunk序列需要5次 DiT
forward，成本约为 last-target 的5倍。Medium/Long 阶段更合理的折中是每个窗口
抽样短、中、长1–3个 target horizon。

### 当前 Short 训练

初始化：

```text
A: re10k-all-a-full-ebs4-from-infinite/full-final.pt
B: re10k-all-b-full-ebs4-from-infinite/full-final.pt
```

不是重新从原始 InfiniteWorld 开始，也不在 A/B 之间互相加载。

训练参数：

| 配置 | 值 |
| --- | --- |
| Dataset | SpatialVID |
| History | 1–2 chunks |
| Target | 最后1 chunk |
| Train scope | Full parameter |
| Optimizer | AdamW |
| LR | `1e-5` |
| Precision | bf16 |
| Micro batch | 1 |
| Gradient accumulation | 4 |
| Effective batch | 4 |
| Max steps | 20,000 |
| Plateau start | 2,000 steps |

配置见 `NAV/config/train_spatialvid_short.yaml`。

### Medium/Long 的推荐样本策略

- Medium 4–7 chunks：在线递归读取全部 history，随机监督1–2个 horizon；
- Long ≥8 chunks：优先使用 DL3DV 等长 episode，每条监督短、中、长 horizon；
- 长链可使用 truncated BPTT，但 detach 边界必须记录；
- 数据集先等权或显式加权，再在数据集内部 shuffle，避免 SpatialVID 约95%的
  帧数占比淹没其他域；
- Validation 必须按 scene/episode 隔离，不能让同一视频不同窗口跨 split。

### 必须记录的实验字段

每次训练日志和 checkpoint 至少包含：

```text
source checkpoint
dataset version / manifest
window policy
min/max chunks
target policy
teacher forcing / scheduled sampling ratio
detach policy
variant A/B
batch size / accumulation
optimizer / LR
steps / plateau rule
```

数据侧定义见 `../02_data/data_preparation_schema_and_status.md` 和
`../02_data/data_preparation_schema_and_status.md`。


## V0 Register 流式世界模型：RE10K 数据与训练记录


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-002` |
| 类型 | 实验记录（Experiment Record） |
| 状态 | Historical Baseline |
| 更新时间 | 2026-07-28 |
| 职责 | 保存 RE10K 上 A/B Register 方案的配置、结果和修订历史 |

### 1. 数据选择

Infinite-World 论文使用两类未公开数据：

- 超过 30 小时的网页第一人称与探索视频，用于预训练；
- 作者用 iPhone 17 Pro 自采的 30 分钟 Revisit-Dense Dataset（RDD），用于长程
  回访记忆微调。

截至 2026-07-24，官方仓库和 Hugging Face 页面均未提供这两个训练集或训练
脚本，因此无法按文件级别复现原数据。按照本项目约定，NAV 改用公共目录已有的
RealEstate10K（RE10K）：

```text
/sharedata/RealEstate10K
```

RE10K 当前本地数据包含 197 个有效帧场景、28,219 张匹配图像和逐帧 3×4
相机外参。NAV 派生了 89 个长度为 162 帧的两 chunk 训练窗口，覆盖 57 个
场景：

```text
/sharedata/RealEstate10K/nav_register/manifests/train.jsonl
```

RE10K 没有键盘动作。`prepare_re10k_manifest.py` 先用 SVD 将旋转矩阵投影到
最近的 SO(3)，再从相邻相机中心与相对旋转计算平移、yaw 和 pitch。它按场景
中位运动尺度归一化，映射到 Infinite-World 的 10 类 `move` 和 10 类 `view`；
极小变化标为 `no-op`，异常大变化标为 `uncertain`。这是伪标签
（pseudo-label），不是 RE10K 原生标注。

### 2. 网络改造

HPMC 的 `latent_encoder.*` 共 68 个 checkpoint keys 被移除。其余
Infinite-World 权重保持不变；审计结果为 905/905 keys 完全匹配、
`missing=0`、`unexpected=0`。

每个已生成或 teacher-forcing 的 chunk latent 先自适应池化成 4×4×4 个
观测 token，再由 16 个持久 Register token 作为 query，通过两层
cross-attention 读取 chunk 信息：

```text
R(k+1) = CrossAttention(query=R(k), key/value=Chunk(k))
```

两个方案共用同一更新器：

- A / `latent_prefix`：维护固定 `[B,16,4,H,W]` 的 latent-like Register。
  `T=4` 只表示4个 Register planes，它不是 VAE 视频 latent。每个空间位置
  的 Register 通过 cross-attention 读取 chunk 在同一空间位置的时间序列，
  随后沿时间轴放在一个 local-memory plane 前；
- B / `dit_condition`：将 Register 投影成 4096 维条件 token，追加到 UMT5
  context，使每个 DiT block 通过原有 cross-attention 读取长期状态。

当前训练冻结 Wan2.1/Infinite-World 1.3B backbone，只更新 Register 更新器和
注入投影。这不是全参数微调；目的是先稳定新记忆通路，并控制显存和灾难性遗忘。

### 3. 数据缓存

Wan2.1 VAE 已将 10 个真实 RE10K 样本编码为两段 latent，每段形状为
`[1,16,21,56,112]`，对应 81×448×896 RGB 帧：

```text
/sharedata/RealEstate10K/nav_register/latents
```

扩展缓存：

```bash
cd /mnt/pool1/sharehome/xiewenyuan
PYTHONPATH=academic/3d_wm_vln/Infinite-World \
CUDA_VISIBLE_DEVICES=0 \
academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python \
academic/3d_wm_vln/NAV/scripts/cache_re10k_latents.py --max-samples 89
```

### 4. 已验证结果

两个方案都完成了真实 RE10K 上的端到端单步：

| 方案 | 目标函数 | Step 1 loss | Register grad norm | NAV 进程峰值显存 |
| --- | --- | ---: | ---: | ---: |
| A latent prefix | Infinite-World RFlow diffusion loss | 0.101113 | 0.171875 | 28.748 GiB |
| B DiT condition | Infinite-World RFlow diffusion loss | 0.389421 | 15.6875 | 13.866 GiB |

loss 不可直接横向判断方案优劣：两次随机 timestep 相同，但注入位置、可训练投影
和有效条件长度不同。这里只用于确认完整 forward/backward、梯度和显存。

正式实验各 1000 steps，10 个 cache 轮换，每 100 steps 保存一次：

| 方案 | GPU | tmux 会话 | 日志目录 |
| --- | ---: | --- | --- |
| A | 0 | `nav_reg_a` | `NAV/log/re10k-latent-prefix-formal-v2` |
| B | 1 | `nav_reg_b` | `NAV/log/re10k-dit-condition-formal-v2` |

训练采用论文相同的 448×896（480P bucket）、AdamW、常数学习率 `1e-5`、
bf16 和 RFlow 配置；当前 batch size 为 1，冻结 backbone，这两项不同于论文
的 16×H800 全参数训练。

查看状态：

```bash
tmux attach -t nav_reg_a
tmux attach -t nav_reg_b
tail -f NAV/log/re10k-latent-prefix-formal-v2/train.log
tail -f NAV/log/re10k-dit-condition-formal-v2/train.log
```

TensorBoard：

```bash
tensorboard --logdir \
  /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log \
  --port 6007
```

### 5. 当前限制

- RE10K 多为房产漫游，域分布比 Infinite-World 的开放域数据窄；
- 动作来自相机位姿伪标签，尺度不确定性和运动模糊会产生噪声；
- 目前只缓存 10/89 个窗口，正式比较前应缓存全部窗口并建立 scene-level
  train/validation split；
- 本历史实验的 Register 从 learnable initial state 开始，在两个 chunk 间
  递归；该结构已被新的 Stage One“首Chunk Extractor、后续Updater”设计取代，
  但旧checkpoint仍按原结构保存。更长的4/8/16 chunk curriculum需要同一场景
  的连续窗口串接；
- 空文本条件用于避免额外加载 11 GB UMT5。后续可对 RE10K 生成 caption cache，
  再恢复文本条件。

### 6. Register 阶段完成结果与全参阶段

两个 Register-only 实验均完成 1000 steps。由于 diffusion timestep 随机，
单点 loss 不适合判断趋势，以下比较前后各 100 steps：

| 方案 | 前 100 mean / median | 后 100 mean / median | mean 变化 |
| --- | --- | --- | ---: |
| A latent prefix | 0.13738 / 0.10039 | 0.15463 / 0.10405 | +12.55% |
| B DiT condition | 0.22996 / 0.18715 | 0.17155 / 0.11778 | -25.40% |

B 有明确改善；A 的中位数基本不变，mean 受高 timestep 的 loss 尖峰影响反而
上升。两者 Register gradient 均下降但未消失。

训练入口已支持 `--train-scope full`，并将上游旧式 reentrant gradient
checkpoint 改为 NAV 内部的 non-reentrant 调用，解决 DiT block 中
`hist/hist_cross` cache 在全参反传时重复访问计算图的问题。

B 全参单步已通过：

- trainable parameters：约 1.42B；
- loss：0.087897；
- gradient norm：1.796875；
- NAV 进程峰值显存：19.668 GiB。

A 全参反传实测需要约 39.49 GiB NAV 显存；GPU 0 同时有约 7.72 GiB 的其他
用户进程，最终因不足 194 MiB OOM。因此当前正式全参阶段先运行 B，A 等待独占
GPU 或引入 FSDP / activation offload，不把降低分辨率的结果与 480P 主实验混用。

### 7. 2026-07-24 结构修订

早期 A 使用 `16×256` token 再广播为空间常数 latent plane，且兼容层会重复
追加 local frame。该结构已经废弃，旧 A checkpoint 不再与新 A 兼容。

当前统一规则：

- local memory 与 Infinite-World 相同，取 history 最后一个 VAE latent
  time step，形状 `[B,16,1,H,W]`，且只拼接一次；
- A 的 history 完全由固定 `[B,16,4,H,W]` Register 代替，DiT latent
  输入顺序为 `[4 register planes; 1 local plane; noisy target]`；
- B 的 latent 侧只有 `[1 local plane; noisy target]`，Register 仅作为
  DiT cross-attention condition；
- Infinite-World 原始推理未指定 `memory_is_precomputed` 时仍走原 HPMC
  和 latest-frame 路径，不受 NAV 接口影响。

新结构单步端到端验证：

| 方案 | Condition latent budget | loss | Register grad norm | 峰值显存 |
| --- | --- | ---: | ---: | ---: |
| A | 4 Register + 1 local | 0.097775 | 0.104004 | 18.533 GiB |
| B | 1 local；Register 在 condition | 0.431135 | 15.375 | 10.243 GiB |

#### B condition 的最终定义

B 不保留 HPMC history。Register 与文本 condition 拼接：

```text
latent side:    [1 latest local plane; noisy target]
condition side: [text tokens; Register tokens]
```

RE10K 本地 manifest 不含 caption、title 或 description，因此 text 使用 UMT5
对空字符串 `""` 的真实编码。Register 从 `[B,16,256]` 投影为
`[B,16,4096]`，追加到512个文本位置之后。拼接结果共同经过 Infinite-World
原 `text_embedding: 4096→1536→1536`，供30层 DiT cross-attention 使用。

### 8. 新 A 全参数正式训练

确认 A 相对 Infinite-World 的生成主干只替换 history memory：

```text
Infinite: [HPMC history; latest local; noisy target]
NAV-A:    [T=4 Register; latest local; noisy target]
```

ActionEncoder、20-channel patch input、condition mask、30层 DiT、文本
cross-attention、velocity head 和 RFlow loss 均沿用 Infinite-World。训练时仅将
上游 reentrant checkpoint 换为数值等价的 non-reentrant checkpoint，以支持
全参反向传播。

正式训练从 `/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt`
开始，加载审计 `missing=0, unexpected=0`；不加载任何旧 NAV-A Register
checkpoint，新 `[B,16,4,H,W]` Register 随机初始化。

```text
tmux session: nav_full_a
run: re10k-new-a-full-from-infinite-formal
trainable parameters: 1,421,435,008
steps: 1000
save interval: 100
```

已完成的前三步 loss 为 `0.097775、0.074608、0.081861`，NAV 进程峰值显存
为 `32.867 GiB`。

### 9. 新 B 全参数正式训练

B 从同一份原始 Infinite-World checkpoint 开始，不加载旧 Register 权重：

```text
latent:    [1 latest local memory; noisy target]
condition: [UMT5 empty-text embedding; 16 Register tokens]
history:   none
```

RE10K 不含 caption，因此空文本由 UMT5-XXL 对 `""` 实际编码并缓存于
`/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt`，形状为
`[1,1,512,4096]`，有效文本 token 数为1。Register condition 为
`[1,1,16,4096]`，两者沿 token 维拼接。

```text
tmux session: nav_full_b
run: re10k-new-b-full-from-infinite-formal
trainable parameters: 1,424,037,200
steps: 1000
save interval: 100
```

全参 step 1 已完成：loss `0.208787`，全参数 gradient norm `18.625`，NAV
进程峰值显存 `16.480 GiB`。

### 10. 公平对照重启：Effective Batch Size 4

RE10K 当前实际缓存训练量：

| 范围 | 窗口 | 场景 | 窗口时长求和 | 去重覆盖时长 |
| --- | ---: | ---: | ---: | ---: |
| 已缓存 | 10 | 7 | 53.714 秒 / 0.0149 小时 | 45.939 秒 / 0.0128 小时 |
| 全部 manifest | 89 | 57 | 479.262 秒 / 0.1331 小时 | 395.439 秒 / 0.1098 小时 |

物理 batch 4 在 B 的全参 backward 中 OOM。修复上游 batch>1 时 action
embedding 被错误广播成 `B²` 的问题后，物理 batch 2 验证通过。因此两组统一
使用有效 batch 4：

| 方案 | Micro batch | Accumulation | Effective batch | Step 1 loss | NAV 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 4 | 4 | 0.073745 | 29.209 GiB |
| B | 2 | 2 | 4 | 0.221912 | 23.375 GiB |

两者均重新从原始 Infinite-World checkpoint 启动，`missing=0`、
`unexpected=0`，不加载旧 Register 权重，并使用相同的 UMT5 empty-text
embedding。计划均为1000个 optimizer steps，每步处理4个样本。

### 11. 使用完整 RE10K train split

训练数据已扩展为本机公共目录中全部180个 train episodes，共生成269个
两-chunk训练窗口：

| 项目 | 数量 |
| --- | ---: |
| Train episodes / scenes | 180 |
| 训练窗口 | 269 |
| 原始长度足够、直接切窗 | 146 |
| 短 episode、全时间跨度均匀重采样 | 123 |
| Latent cache | 269 |

长 episode 采用162帧窗口、stride 81，并补充最后一个窗口覆盖尾部。短于162帧
的 episode 不再丢弃，而是在其完整时间跨度上均匀采样到162帧；重复位置的
action label 为 no-op。全部 cache 已逐文件检查，manifest 与 cache ID
一一对应，`missing=0`、`extra=0`、坏文件为0；tensor shape 均为
`[1,2,16,21,56,112]`。

正式运行名为：

```text
A: re10k-all-a-full-ebs4-from-infinite
B: re10k-all-b-full-ebs4-from-infinite
```

两组均从原始 Infinite-World checkpoint 初始化，全参数训练1000个 optimizer
steps，不加载旧 Register 权重。由于共享 GPU 上存在其他任务，两组最终均使用
`micro batch=1, gradient accumulation=4, effective batch=4`，并开启
PyTorch expandable-segments allocator 以降低共享显存碎片。


## V0 Stage One 1.0：DL3DV 从头训练实验


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-005` |
| 类型 | 训练实验规范（Training Experiment Specification） |
| 状态 | Historical Baseline / 旧入口已从当前代码删除 |
| 更新时间 | 2026-08-07 |
| 职责 | 固定 Stage One 1.0 的初始化、数据、采样、显存实测和运行入口 |

### 实验定义

本文件保留 V0/early-V1 历史实验记录。文中的旧运行命令不再是当前 NAV
可执行入口；如需复现，请从 git 历史恢复清理前版本。

Stage One 1.0 分别训练 A/B（DEC-019，2026-08-03 更新）：

- Backbone 从原始 Infinite-World checkpoint 初始化；
- 不加载既有 RE10K A/B checkpoint，也不加载旧 A/B-10000 checkpoint；
- HPMC history 移除，Local Memory、Action/Text和RFlow保持不变；
- 新的 Register Extractor/Updater 随机初始化；
- Wan DiT、Extractor、Updater 全参数训练；
- 数据只使用 DL3DV full-episode latent 与 dense Action；
- 文本使用 empty UMT5 embedding；
- 训练 **1000 effective optimizer steps**，effective batch size 16；
- 每 **100 effective steps** 保存一次完整 checkpoint；
- 不使用 plateau early stop。

DEC-018 的 10000-step 计划已被 DEC-019 取代；旧 `stage-one-v1-dl3dv-{a,b}-10000`
run 作为历史保留，不用于正式续训。

### A 停止与续训记录（2026-08-04）

- 2026-08-04 18:59，A（`latent_prefix`，cuda:0）按用户指示优雅停止
  （SIGTERM，进程 4023980 已干净退出）。
- 停止时 A 已训练到 **step 541**，最后保存的 checkpoint 为
  `full-step-000500.pt`；step 501–541 的进度已写入 `train.log` 但未落盘
  到 checkpoint。
- B（`dit_condition`，cuda:1）未受影响，继续运行。
- **后续 A 从 step 500 续训**：使用 `full-step-000500.pt` 作为
  `resume_checkpoint`，run 名沿用
  `stage-one-v1-from-scratch-a-ebs16-1000`，其余参数不变（micro=1、accum=16、
  effective=16、total 1000 effective steps、save-every 100、DL3DV full-episode
  latent、empty UMT5）。续训命令示例：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  latent_prefix 0 1 1000 stage-one-v1-from-scratch-a-ebs16-1000 100
```

并在对应配置中设置 `resume_checkpoint=
log/stage-one-v1-from-scratch-a-ebs16-1000/full-step-000500.pt`，
`register_checkpoint` 同源（若脚本区分）。续训前确认 GPU0 显存可用且
`full_episodes_v1` latent prep 是否仍在 GPU0 占用显存。

### Register 语义

首个 History Chunk：

\[
R_1=\operatorname{Extractor}(Z_0)
\]

后续 History Chunk：

\[
R_{i+1}=\operatorname{Updater}(R_i,Z_i)
\]

A 的 Extractor 将首 Chunk 时间池化为4个 spatial planes，再通过独立
Cross-Attention/FFN 提取 `[B,16,4,H,W]` Register。

B 的 Extractor 将首 Chunk 池化为 `4×2×2=16` 个观测 token，再通过独立
Cross-Attention/FFN 提取 `[B,16,256]` Register。

两者均不包含 `initial_registers` 参数；Extractor 与 Updater 不共享参数。

### 训练样本

141条 DL3DV full episode 全部至少包含4个 chunks，均可参与训练。

每个 optimizer step 从以下 history 长度中按打乱循环等比例选择：

```text
1 history: C0       → C1
2 history: C0,C1    → C2
3 history: C0,C1,C2 → C3
```

- episode 顺序每个 epoch shuffle；
- 每条 episode 内随机选择连续窗口起点；
- 只对最后一个 target 计算 Diffusion/RFlow loss；
- Register 在 forward 内在线 Extract/Update，不落盘；
- physical batch 内使用相同 history length，避免不规则张量 padding。

### 显存实测

设备为单张约47.40 GiB GPU，bf16、全参数、gradient checkpointing。

DEC-019（ebs16）实测：

| Variant | Micro batch | Accumulation | Effective batch | NAV 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| A latent_prefix | 1 | 16 | 16 | 33.24 GiB |
| B dit_condition | 2 | 8 | 16 | 28.71 GiB |

DEC-016/017（旧 ebs4/bs1 probe）实测保留供追溯：

| Variant | Physical batch | 结果 | 显存 |
| --- | ---: | --- | ---: |
| A | 4 | OOM during backward | 46.82 GiB已占用，仍需772 MiB |
| A | 1 | Pass | 29.21 GiB allocated peak |
| B | 1 | Pass | 16.48 GiB allocated peak |

DEC-019 调度采用：

- GPU 0：A，micro=1、accum=16、effective=16，1000 effective steps；
- GPU 1：B，micro=2、accum=8、effective=16，1000 effective steps；
- 两卡其余显存继续并行准备 `full_episodes_v1` latent（7-shard 幂等续传）。

2026-07-30启动时GPU 1被其他账号的 `vlm_server.py` 和 `depth_server.py` 占用
约41 GiB，不能擅自终止。该情况在 DEC-019 重启时已重新评估；当前 B 的
ebs16 配置在 GPU1 上稳定运行。

### 配置与入口

### 项目内 `from scratch` 的统一定义

本项目后续实验所称的 `from scratch` 并非随机初始化整个视频生成模型，而是：

- 仅加载原始 Infinite-World checkpoint 中的 Wan/DiT 预训练权重；
- 加载后按方案 A 或 B 改造网络结构；
- Extractor、Updater/Register 等新增模块随机初始化；
- 不加载任何旧 RE10K、旧 A/B 或其他 NAV 训练 checkpoint；
- `resume_checkpoint=null` 且 `register_checkpoint=null`。

因此，`from scratch A/B` 表示两种结构均从同一个原始 Infinite-World 基线独立
开始训练，而不是 A/B 互相继承，也不是续训此前实验。

配置：

```text
NAV/config/train_stage_one_v1_dl3dv.yaml
```

统一入口：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  <latent_prefix|dit_condition> <gpu> <physical_batch> <steps> <run_name> <save_every>
```

DEC-019 正式单任务示例（effective batch 16、1000 effective steps、每 100
effective steps 存 checkpoint）：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  latent_prefix 0 1 1000 stage-one-v1-from-scratch-a-ebs16-1000 100
```

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  dit_condition 1 2 1000 stage-one-v1-from-scratch-b-ebs16-1000 100
```

注意：`physical_batch` 与 `gradient_accumulation_steps` 由配置中的
`effective_batch_size=16` 反推（A: micro=1 → accum=16；B: micro=2 → accum=8）。

入口自动激活项目 Python 环境，并启用
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

数据余量调度入口：

```text
NAV/scripts/run_latent_shard_queue.sh
```

当前 tmux：

```text
nav_stage1_v1_a
nav_stage1_v1_b
nav_full_episode_prep
nav_stage1_v1_tensorboard
```

正式 A/B ebs16 的独立 TensorBoard 运行在远程端口 6011，不包含旧 1000-step、
10000-step、smoke 或其他实验。

### Stage One 验证边界与后续待补实验（2026-08-07）

本节界定 Stage One 1.0 已验证什么、未验证什么，并给出后续待补实验清单。
对应 `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` 记录 4/7/8 与
`doc/05_insight/insight_log.md` R13–R16。

#### 已验证（Stage One 初步验证成立）

| 项 | 证据 | 强度 |
| --- | --- | --- |
| 训练能简单收敛 | A 训到 step 541（ckpt@500）、B 训到 step 1000，loss 下降无 NaN | 强 |
| Register 在**训练集**上表现正常 | gthist 用 DL3DV **训练 episode** 的 3 GT 历史 chunk 预测第 4 chunk，A@500/B@1000 单步保真 L1=0.0874/0.0917、SSIM=0.54，优于 InfiniteWorld（L1=0.1152）（记录 8） | 中（仅 1 episode、单 seed、空 text） |
| Register 是固定预算、比 HPMC 小 | A=4 帧 latent（~8MB）/ B=16 token（~128KB）vs HPMC 20 帧（~40MB），token 比 21:5:1 | 强（结构事实，非性能） |
| 短程 autoreg 技术质量 | 3-chunk autoreg VBench：A@500 MS/AQ/IQ 正常、DD=true（记录 7） | 中（仅 1 demo 场景） |

#### 未验证（Stage One 尚未证明）

1. **泛化 + action 跟随**：gthist 用训练 episode + GT pose-derived action；autoreg 用
   demo 单场景 + 手录 0001.json。无 held-out 场景、无跨分布 action 跟随证据。
2. **长程有效显存缩减**：Register 尺寸更小已证，但"长程下保持性能"未证——NAV A/B
   的 16+ chunk autoreg 从未跑过（仅 InfiniteWorld 跑过 16-chunk）。
3. **autoreg 等价于 teacher-forced**：gthist 单步保真好，但 autoreg 3-chunk 里
   B 的 DD 塌缩为 false（记录 7）——autoreg ≠ teacher-forced，且无 NAV 长 autoreg 证据。
4. **Register 优于 HPMC**：3-chunk 下 HPMC 的 `T_in=63≤80` 不触发二次压缩，
   与"全量历史"等价，Register 与 HPMC 容量上等价、比不出设计差异（R13）。
   现有 3-chunk 对比对"Register 优于 HPMC"零证据。
5. **text 通道可用**：Stage One 用空 text，UMT5 cross-attention 未训练；
   推理却给真实 prompt。R2R 指令接入路径未定（R15）。
6. **train-test action 一致**：训练 pose-derived 离散 vs 推理手录，分布不一致（R16）。

#### 后续待补实验（按性价比排）

1. **多场景多 seed autoreg 3-chunk**（复用 stats20 框架扩到 NAV A/B）——证伪/确认
   B 的 DD 塌缩是否系统性、A 是否稳定。
2. **NAV A/B 16-chunk autoreg**（用已改好的 cached-latent 脚本）——证明长程下
   固定预算 Register 不崩，是"有效显存缩减"的核心证据。
3. **长程训练**（history 课程扩到 16+ chunk）——当前 updater 仅训过 ≤3 次更新，
   长程下 OOD；要让 16-chunk autoreg 对 NAV 公平，必须先长程训练 updater。
   这是 R13"递归更新 vs 启发式窗口"主张的唯一证据来源。
4. **held-out 场景 + 训练分布 action 的 gthist**——证明泛化 + action 跟随
   （DL3DV 未训练 episode，action 仍用 pose-derived，隔离 text）。
5. **text 通道决策实验**——决定 Stage One/二是否回头引入 caption 联合训练
   （R15），影响 Stage Three 能否讲 R2R 故事。
6. **action 来源统一**——评估 train-test action gap 对结论的污染（R16）。

#### 论文叙事风险

若长程对比（实验 2+3）做不出或 NAV 不优于 HPMC，"固定预算递归更新记忆优于
启发式窗口"这一强主张站不住，需退回到弱主张"Register 是一个可导航的固定
预算记忆"。当前 Stage One 定位为"初步验证"：仅证明 Register 机制能训得动、
训练分布内能工作、结构上比 HPMC 省——仅此而已。


## V0 Stage Two：视频生成前向中的 3D Probe 与监督规范


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-004` |
| 类型 | 训练规范（Training Specification） |
| 状态 | Proposed / Confirmed Semantics / Not Implemented |
| 更新时间 | 2026-07-30 |
| 职责 | 定义完整视频生成前向中的 3D Probe、选层、监督位置与联合损失 |

### 结论

Stage Two 完全保持视频生成训练方式，不把模型改造成独立 3D encoder，也不引入
Navigation Policy。训练仍然包含：

```text
真实 History Chunks
    → Extractor-first / Updater-later Register
    → Register + Local Memory + Action/Text + Noisy Target
    → Wan DiT
    → Diffusion/RFlow loss
```

在这条完整生成前向上额外执行：

```text
多个候选 DiT block 的 Register-after-DiT
    → Frozen diagnostic probe
    → 选择3D最可读的层 l*
    → 3D Teacher/GT supervision
```

Stage Two 的交付物是经过 3D 监督的视频生成模型，以及可供 Stage Three 导航
读取的 `Register-after-DiT @ selected layer` 接口。

### 前置条件：Stage One 必须先改造

Stage Two 不再使用 learnable initial Register。一个 episode 的在线状态为：

\[
R_1=\operatorname{Extractor}(Z_0)
\]

\[
R_{i+1}=\operatorname{Updater}(R_i,Z_i),\quad i\ge1
\]

即：

- 第一个 History Chunk 负责提取 Register；
- 后续 History Chunk 才更新 Register；
- checkpoint 保存 Extractor/Updater 的能力参数；
- checkpoint 不保存 episode-specific Register。

当前旧 A/B checkpoint 仍包含 learnable initial Register，只能作为权重初始化
来源；进入 Stage Two 前必须完成结构迁移与 smoke test。

### 多 Chunk 视频生成样本

从同一 episode 采样：

```text
History: C0, C1, ..., Ck
Generation Target: Ck+1
3D Teacher/GT: C0, C1, ..., Ck 的真实历史几何标签
```

3D GT 只来自已经观察到的 History。未来生成 Target 即使训练数据中存在真实帧，
也不作为主 3D Teacher 对象；它仍只承担原视频生成的 Diffusion/RFlow target。
这样既保持完整视频生成训练，又确保被监督的是用于压缩历史的 Register。

历史使用真实干净 latent：

\[
Z_i=\operatorname{VAE}(C_i)
\]

按 Stage One 规则建立和更新 Register：

\[
R_1=E(Z_0),\qquad
R_{i+1}=U(R_i,Z_i)
\]

目标仍按原 RFlow/Diffusion 方式加噪：

\[
X_{k+1,t}=\alpha_t Z_{k+1}+\sigma_t\epsilon
\]

完整生成预测为：

\[
\hat v_{k+1}
=D_\theta(X_{k+1,t},t,R_{k+1},L_k,A_{k+1},\mathrm{text})
\]

其中 \(L_k\) 是 Infinite-World latest local memory。不能为了做 3D 监督绕过
noisy target、Local、Action/Text 或 DiT 生成 blocks。

### 主要 Probe 对象

Register 在注入 DiT 前只是 history memory。Stage Two 需要定位的是它参与视频
生成计算后形成的 contextualized representation：

\[
\widetilde R_{k+1}^{\,l}
=
\operatorname{SelectRegister}
\left[
D_\theta^{1:l}
(X_{k+1,t},L_k,R_{k+1},A_{k+1},\mathrm{text})
\right]
\]

对多个候选层 \(l\) 读取 \(\widetilde R^l\)，训练轻量 probe，判断哪一层最适合
3D 监督和 Stage Three 导航。

#### A 的读取

A 的 Register 是 latent prefix：

```text
[Register prefix; Local Memory; Noisy Target]
```

每个候选 block 后按固定 token range 切出 Register prefix hidden。必须同时保存
grid size 和 prefix offset，禁止把 Local/Target token 混入 Register probe。

#### B 的读取

B 的 Register 与 UMT5 text 拼接为 condition：

```text
[Text condition; Register condition]
```

必须确认选定 block 后能读取“经过 DiT 更新”的 Register。如果 B 的
cross-attention 仅把 condition 当作静态 K/V，而不回写 condition hidden，则
当前实现不存在严格意义的 `Register-after-DiT`。此时必须增加可回写的
Register stream、双向 cross-attention 或 block-level Register updater，再做
与 A 同口径的 probe。

不能把进入 DiT 前的静态 condition embedding 误称为 Stage Two 导航表示。

### Probe 实验

#### 冻结范围

初始 probe 阶段冻结：

- Wan VAE；
- Wan DiT；
- Extractor；
- Updater；
- Action/Text encoders。

只训练小型 Linear/MLP/Transformer readout，避免 probe 本身重写 backbone。

#### 候选层

从浅、中、深层均匀选择若干 DiT blocks。实际层号必须读取当前 Wan1.3B 代码后
写入 config，不在文档中凭经验硬编码。

每个候选层至少评估：

- Relative Camera Pose；
- Camera Trajectory；
- Coarse Layout / Occupancy；
- VGGT scene/register feature alignment；
- 跨 Chunk geometry retention；
- 3D probe 指标随 history 长度的变化。

可额外 probe target/history spatial tokens 的 Depth、Pointmap 或 patch feature，
用来判断 Register 的3D信息是否来自 DiT 的时空建模，但 Stage Three 的默认接口
仍是 `Register-after-DiT`。

#### 选层标准

\[
l^\*=\arg\max_l
\operatorname{Score}
(\text{3D accuracy},\text{history retention},\text{cost},\text{stability})
\]

选择不能只看单个 depth 或 pose 数字，还要考虑：

- 多 Chunk 增长时是否稳定；
- A/B 是否使用同一公平 protocol；
- 导出该层是否增加额外 DiT 计算；
- 表示尺寸是否适合固定预算导航输入；
- 不同 dataset/scene split 上是否泛化。

### 3D Teacher 与 GT

Frozen VGGT/VGGT-Ω 读取相同样本的干净 RGB，提供：

- Camera Pose / Relative Motion；
- Scene/Register Feature；
- Patch Geometry Feature；
- Depth、Confidence；
- 可用时的 Pointmap / Matching。

已有真实 Pose/Depth 时优先使用 GT。Pseudo-label 必须保留 confidence、
visibility 和 validity mask。

Teacher 只提供 loss target，不作为 DiT condition，避免先验泄漏。

### 时间与空间对齐

每个81-frame Chunk 对应21个 Wan VAE latent time steps。逐点 Teacher target
需要：

1. 将81帧 Teacher 输出划分为21个 temporal bins；
2. 在 bin 内按 confidence 聚合；
3. 将空间网格 resize/project 到 DiT token grid；
4. 保存 source frame indices、bin、grid、mask 和 teacher version。

禁止用未经验证的单帧 `frame_id=4*latent_id` 直接硬对齐。Pose/trajectory head
必须明确预测21-step latent pose、81-frame pose，或 Chunk-level transform，
三者不能混用。

Register-level scene/trajectory supervision不要求与每个 patch逐点对应，但必须
明确参考坐标系和时间覆盖范围。

### 正式联合 3D 监督

Probe 选出 \(l^\*\) 后，在完整视频生成 forward 中加入：

\[
\mathcal L_{\mathrm{reg3D}}
=
\lambda_f\mathcal L_{\mathrm{teacher}}
(P(\widetilde R^{l^\*}),G)
+\lambda_{\mathrm{pose}}\mathcal L_{\mathrm{trajectory}}
+\lambda_{\mathrm{map}}\mathcal L_{\mathrm{layout}}
+\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{history}}
\]

可选 spatial auxiliary loss：

\[
\mathcal L_{\mathrm{spatial}}
=
\lambda_d\mathcal L_{\mathrm{depth}}
+\lambda_p\mathcal L_{\mathrm{point}}
+\lambda_{\mathrm{patch}}\mathcal L_{\mathrm{patch\ feature}}
\]

视频生成主体 loss：

\[
\mathcal L_{\mathrm{wm}}=\mathcal L_{\mathrm{RFlow/Diffusion}}
\]

总损失：

\[
\mathcal L
=
\lambda_{\mathrm{wm}}\mathcal L_{\mathrm{wm}}
+\lambda_{\mathrm{reg3D}}\mathcal L_{\mathrm{reg3D}}
+\lambda_{\mathrm{spatial}}\mathcal L_{\mathrm{spatial}}
\]

\(\lambda_{\mathrm{wm}}\) 必须大于0，保证 Stage Two 仍然是视频生成训练。各 loss
权重通过 gradient norm 和独立 validation 指标校准，不能只按 loss 数值大小
相加。

### 梯度路径

必须验证：

\[
\mathcal L_{\mathrm{reg3D}}
\rightarrow \widetilde R^{l^\*}
\rightarrow \mathrm{DiT}^{1:l^\*}
\rightarrow R
\rightarrow \mathrm{Extractor/Updater}
\]

以及：

\[
\mathcal L_{\mathrm{wm}}
\rightarrow \mathrm{DiT}
\rightarrow \mathrm{Register\ injection}
\]

日志至少记录：

- selected block Register gradient norm；
- Extractor gradient norm；
- Updater/Cross-Attention Q/K/V gradient norm；
- 3D projector/head gradient norm；
- Diffusion、Register 3D、Spatial auxiliary 各 loss；
- Teacher valid/confident ratio；
- 各候选层 probe 指标；
- 生成质量与3D指标的变化。

如使用 truncated BPTT，必须记录 detach 的 Chunk 边界。

### 分阶段执行

#### Phase 0：Stage One 结构迁移

- 实现 Extractor-first / Updater-later；
- 从旧 A/B checkpoint 迁移可复用权重；
- 验证 episode reset 无场景状态泄漏；
- 验证2、3及更多 Chunk 的梯度路径。

#### Phase 1：Frozen layer probe

- 完整运行视频生成 forward；
- 缓存多个 block 的 Register-after-DiT；
- 冻结 backbone，只训练 probe；
- 在独立 scene split 上选择 \(l^\*\)。

#### Phase 2：3D head warm-up

- 固定 \(l^\*\)；
- 训练3D projectors/readout；
- 可先冻结大部分 DiT；
- 始终保留 Diffusion/RFlow loss。

#### Phase 3：Full-parameter 3D-aware generation

- 解冻 DiT、Extractor 和 Updater；
- 延续 Short/Medium/Long history curriculum；
- 联合优化 diffusion 与 3D losses；
- 监控生成质量—3D能力 Pareto trade-off。

到此 Stage Two 结束。Navigation Head/Policy 的训练属于 Stage Three。

### 必须包含的对照

| 对照 | 目的 |
| --- | --- |
| Stage One，无3D loss | 判断3D监督收益 |
| 各候选 DiT layer frozen probe | 选择 \(l^\*\) |
| Register-before-DiT vs Register-after-DiT | 验证DiT contextualization价值 |
| Register probe vs spatial-token probe | 确认3D信息分布 |
| 只用 Register 3D loss vs 加 spatial auxiliary | 判断辅助稠密监督价值 |
| A vs B | 比较 latent-prefix 与 condition Register |
| GT Pose vs VGGT pseudo-label | 判断Teacher噪声影响 |

### Stage Two 验收标准

1. Stage One 已完成 Extractor-first / Updater-later 改造；
2. A/B 均有明确、可复现的 Register-after-DiT 读取规则；
3. Frozen probe 找到显著优于随机/注入前基线的3D层；
4. 选定层在不同 history length 和 scene split 上稳定；
5. 3D loss 梯度能到达 DiT、Extractor 与 Updater；
6. Diffusion/RFlow loss 始终参与训练；
7. 3D联合训练后的生成质量退化处于预设范围；
8. 输出 selected layer、tensor shape、normalization、projector与checkpoint。

总体网络边界见
`../01_model/model_evolution_and_current_architecture.md`；Stage Three 导航接口见
`../01_model/model_evolution_and_current_architecture.md`。


## V0 多数据集增量预处理与分阶段训练


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-001` |
| 类型 | 训练规范（Training Specification） |
| 状态 | Historical Baseline / 旧入口已从当前代码删除 |
| 更新时间 | 2026-07-29 |
| 职责 | 定义增量 latent、shuffle 和 Short/Medium/Long 课程训练 |

训练样本中 Register 的在线递归、teacher forcing、last-target 与推理差异，以
`training_plan_and_experiment_log.md` 为唯一事实来源；数据 tensor 和
Action 的定义分别以 `../02_data/data_preparation_schema_and_status.md` 与
`../02_data/data_preparation_schema_and_status.md` 为准。

本文件保留 V0 多数据集课程训练历史。文中的旧配置和训练脚本已从当前 NAV
代码主线删除，不再作为 V1 训练入口。

### 数据隔离原则

DL3DV、SpatialVID、Argoverse 2 和 RE10K 的下载目录只读使用。所有派生产物统一写入：

```text
/sharedata/NAV/derived/
├── manifests/episodes.jsonl
└── latents/<dataset>/<sample_id>.pt
```

`prepare_multidataset_manifest.py` 每次执行都会重新扫描当前完整 episode，并使用
临时文件原子替换 manifest。因此下载中新增加的数据会在下次执行时自动出现；未完成
episode 不进入 manifest。脚本不会移动、重命名或修改下载文件。

`cache_multidataset_latents.py` 以 `dataset/sample_id` 和 sidecar 中的
`cached_chunks` 判断完成度。Short 阶段最多写入 3 chunks；Medium 只补第
4–7 chunks；Long 只补第 8 chunk 以后的内容。更新使用临时文件原子替换，已经
编码的 chunks 不重复计算。RE10K 已有完整 latent 使用硬链接复用。

以下是2026-07-27旧统一流水线的历史 manifest，共4,327条；不代表当前
SpatialVID Short 的数据总量：

| 数据集 | Episode |
| --- | ---: |
| RE10K | 269 |
| DL3DV | 50 |
| SpatialVID | 3,997 |
| Argoverse 2 | 11 |

Kinetics 当前只有 episode 级 VGGT 运动统计，没有完整逐帧外参，因此暂不伪造
`move/view`，也不进入本轮有 Pose 的训练 manifest。

### 统一格式

每条 manifest 保存视频或图像源、所选帧、离散 `move/view`、帧数和 chunk 数。
每个 chunk 固定 81 帧，对应 Wan VAE 的 21 个 latent 时间位置。

- DL3DV：将 `transform_matrix` 的 camera-to-world 转为 world-to-camera；
- SpatialVID：使用官方 `poses.npy` 的 world-to-camera，并用 Slerp 将 60 个稀疏
  Pose 插值到原视频帧；
- Argoverse 2：将 `city_SE3_egovehicle` 与前向相机外参组合，使用
  `ring_front_center`；
- RE10K：复用既有 GT Pose 生成的 `move/view`。

### 增量预处理命令

```bash
bash NAV/scripts/run_incremental_data_prep.sh cuda:1 0 1
```

多 GPU 时第二、三个参数分别是 shard index 和 shard 总数。新增下载完成后重复同一
命令即可，只会编码新增样本。

### Shuffle 与长上下文窗口

训练脚本不再按文件排序固定循环。它使用固定 `shuffle_seed`：

1. 每个 epoch 洗牌全部合格缓存；
2. 每个 batch 在允许范围内随机选择连续 chunk 数；
3. 每条 episode 随机选择窗口起点；
4. 窗口最后一个 chunk 是 diffusion target；
5. 之前所有 chunk 按时间顺序递归更新 Register；
6. local memory 始终取目标前最新 chunk 的最后一个 latent frame。

因此 6-chunk episode 在短阶段也可以贡献随机 2–3 chunk 片段，而不是只能进入
中等阶段。

### A/B 三阶段 Curriculum

#### DL3DV history-extension 测试

此前基于 `images_8` 稀疏 SfM 帧得到的3–5 chunk口径已废止。canonical v1
改用完整原视频固定分块缓存；DL3DV 当前146条原视频平均约52.97 chunks。
只有 `full_episodes_v1` latent 与 dense Action 完成后，才启动6-chunk以内的
A/B history-extension 测试。

#### 当前优先阶段：SpatialVID Short

在进入原三阶段多数据集课程前，先执行 SpatialVID 的 2–3 chunk
short-history 训练。A/B 不再从原始 InfiniteWorld 初始化，而是分别继承已经
完成的 RE10K 1000-step 全参数 checkpoint：

```text
A: NAV/log/re10k-all-a-full-ebs4-from-infinite/full-final.pt
B: NAV/log/re10k-all-b-full-ebs4-from-infinite/full-final.pt
```

每条落盘 SpatialVID episode 以 episode ID 的稳定哈希选择一个连续窗口：

```text
2-chunk: [teacher-forced C0] → target C1
3-chunk: [teacher-forced C0, C1] → target C2
```

离线并行准备两类相互独立的产物：

```text
/sharedata/NAV/derived/latents/spatialvid/<sample_id>.pt
/sharedata/NAV/derived/actions/spatialvid_short/<sample_id>.json
```

VAE latent 与逐帧 `move/view` 可以并行计算；Register 不落盘。训练 forward
从 checkpoint 中的 learnable initial state 开始，依次用干净 history latent
在线更新 Register，local memory 取最新 history chunk 的最后一个 latent
frame，只对窗口末尾 target 做一次 RFlow diffusion loss。chunk 之间不
detach，因此 target loss 会穿过全部 Register updates 反传。

以上描述仅对应当前尚未迁移的历史训练代码。新的 Stage One 结构必须改为首个
History Chunk 经 Extractor 产生 Register，后续 Chunk 才经 Updater；迁移后
课程数据与 last-target 口径不变。

本阶段采用 `random-prefix next-chunk teacher forcing` 的 last-target 版本，
而不是在训练中完整 diffusion rollout 历史。配置唯一事实来源为
`config/train_spatialvid_short.yaml`，持久后台入口为：

```bash
tmux new-session -d -s nav_spatialvid_parallel \
  "bash NAV/scripts/run_spatialvid_short_parallel_pipeline.sh"
```

流水线以7个稳定shard并行准备 latent（GPU 0三个、GPU 1四个），配对校验通过
后释放VAE worker，并在GPU 1上依次执行A、B。两者均为全参数训练、effective
batch size 4、最多20,000 steps，并允许loss平台后提前结束。

下表保留原三阶段 curriculum 目标。其旧 Stage 1“从原始 InfiniteWorld 初始化”
已被当前 SpatialVID Short 决策替代：当前 A/B 必须分别从各自 RE10K 1000-step
checkpoint 继续；两者仍不互相加载权重。Medium/Long 的初始化策略需在 Short
结果确认后冻结为新配置。

| Curriculum Phase | 连续片段 | 数据资格 | 最大步数 | 转换条件 |
| --- | --- | --- | ---: | --- |
| Short | 2–3 chunks | 至少 2 chunks | 20,000 | loss 进入平台 |
| Medium | 4–7 chunks | 至少 4 chunks | 15,000 | loss 进入平台 |
| Long | ≥8 chunks | 至少 8 chunks | 10,000 | loss 进入平台或到上限 |

“几乎收敛”定义为：至少训练规定最小步数后，以 200 step 平均 loss 为窗口；连续
3 个窗口的相对改善不足 1% 时进入下一阶段。Medium/Long 加载上一阶段的完整
backbone 和 Register，优化器在阶段边界重新初始化。

以下原课程数量为 2026-07-27 的历史基线；在 SpatialVID Short 完成后重新统计：

- Short 可用：4,327；
- Medium 可用：3,101；
- Long 可用：1,409。

单独调试某一完整课程时可使用：

```bash
bash NAV/scripts/run_curriculum_train.sh latent_prefix 0 experiment_a
bash NAV/scripts/run_curriculum_train.sh dit_condition 1 experiment_b
```

正式实验的唯一推荐入口是分阶段后台流水线：

```bash
tmux new-session -d -s nav_staged_pipeline \
  "bash NAV/scripts/run_staged_background_pipeline.sh"
```

执行依赖为：

```text
Short latent ready
├── GPU 0：A / Short
└── GPU 1：扩展 Medium latent
             ↓
          扩展 Long latent
             ↓
          B / Short → Medium → Long
```

A 在各阶段等待对应 ready marker，B 在完整 latent 后使用 GPU 1。任何 latent
缺失或编码错误都会触发增量重扫，不会用不完整数据静默训练。状态与日志统一位于
`NAV/log/full_pipeline/`。

## 2026-08-17 Formal V1 收敛 Gate：先接口，后结构

当前目标不是重新发明一个“看起来能跑”的轻量路径，而是在已经证明可收敛的
IW-aligned cotrain 上逐步替换为最终 V1 所需的数据形式、action interface 和
Register/action routing。验收顺序固定如下：

1. **Combo-schema baseline gate**
   - 结构保持旧版快速收敛路径：
     `register_mode=legacy_extract_update`，
     `current_action_mode=legacy_iw_move_view`。
   - 数据/接口切到最终 action schema：
     `combo_id = trans_id * 12 + rot_id`，`combo_dim=144`。
   - Stage One/Two 中 `A_noise`、shared action tokens、`combo_logits`
     必须存在，但 action loss 为 0；Stage Three 使用
     `CE(combo_logits, action_combo)`。
   - 当前运行：
     `log/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216/`
   - TensorBoard 端口：`6026`。
   - 已确认 preflight 字段：
     `a_hist_combo=[1,10]`，`a_cur_combo=[10]`，
     `a_noise=[10,6]`，`pose_target=[4,9]`。
   - 早期曲线：
     step100 `train/loss_visual=0.1793`，`train/loss_pose=0.0031`。
     这与旧 StageOne step100 `loss=0.1680` 接近，速度约
     `11.1 s/step`，与旧 cotrain 约 `11.4 s/step` 接近。
     但 `eval@100 visual=0.5723` 偏高；由于当前 eval batch 很小且窗口随机，
     不能单独作为失败结论，需等待 500/1000 step 和 last10 统计。

2. **Formal Register gate**
   - 只切 `register_mode=fixed_rnull_unified`。
   - `A_cur` 仍走 `legacy_iw_move_view`，避免把 Register 改动和
     action routing 改动混在一起。
   - 目标：训练速度、video convergence、pose convergence 不显著退化。

3. **Formal A_cur routing gate**
   - 在 fixed-rnull 已通过后，只切
     `current_action_mode=shared_tail_token`。
   - 目标：关闭 IW native move/view embedding 后，A_cur combo token 仍能支撑
     video cotrain 收敛，并保持 Stage3 policy token 接口一致。

后台守护脚本：

```bash
tmux new-session -d -s nav_formal_v1_gates_after_combo \
  "cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV && bash scripts/watch_combo_schema_then_formal_v1_gates.sh"
```

该 watcher 会先等待 combo-schema baseline 到 1000 step，然后按阈值检查：

```text
baseline:  last10 train visual mean <= 0.16, last eval visual <= 0.60
candidate: last10 train visual mean <= 0.18, last eval visual <= 0.60
```

这里 `last eval visual` 只是灾难检查，不作为精细排序指标；当前
`RE10KStage2Dataset.__getitem__` 在 eval 时仍随机抽 `window_start`、
`z_future_noise` 和 `visual_timestep`，因此 eval 曲线有明显随机性。正式比较
生成质量时应使用固定 seed/window/noise 的独立 evaluator 或 decoded-video 指标。

2026-08-17 19:43 起，watcher 已修正为等待对应 `step_*.pt` checkpoint
真正落盘后再启动下一 gate，避免 train row 到 1000 后仍在 eval/save 时抢占
GPU。此外每个 gate 的 checkpoint 完成后都会调用 fixed-seed evaluator：

```bash
python scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py \
  --checkpoint <run>/checkpoints/step_001000.pt \
  --max-batches 32 \
  --eval-seed 20260817
```

该 evaluator 显式固定 torch RNG；dataset window RNG 来自 checkpoint
`train_config.seed`，DataLoader 使用 `shuffle=False`，因此比训练脚本内置 eval
更适合做跨结构对比。

若任一 gate 不通过，watcher 会停止，不会自动推进到更“最终”的结构，避免产生
错误通过结论。报告位置：

```text
result/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gates/
log/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gate_watcher_iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216.log
```

### Stage3 变量接口 preflight（真实 R2R 数据）

为了确认前两阶段保留的 action noise/output token 不是“孤立支路”，Stage3
训练脚本加入了 `--preflight-data-only`，只构造真实 R2R policy dataset 并写出
字段形状，不加载 Wan/IW 大模型，不占 GPU。该检查只验证数据和 schema，不作为
训练效果证明。

运行记录：

```text
run = log/v1_stage3_iw_aligned_r2r_policy/stage3_combo_schema_preflight_20260817_1933/
register_mode = fixed_rnull_unified
current_action_mode = shared_tail_token
dataset_size = 32  # 当前默认 smoke obs-latent manifest；full manifest 准备好后同脚本切换路径
```

关键字段：

```text
history_latents = [4,16,1,56,112]
a_hist_combo    = [4,10]
z_obs           = [16,1,56,112]
a_noise         = [10,6]
action_target   = [10,6]  # 兼容旧 continuous head/ablation
action_combo    = [10]
combo_dim       = 144
policy_output   = combo_logits=[B,H,144]
loss            = CE(combo_logits, action_combo)
```

结论：最终第三阶段所需变量可以从当前 R2R 数据接口构造出来；Stage One/Two 中
同一 `A_noise`/shared action hidden/`combo_logits` 路径保持存在但 action loss
为 0，Stage Three 才对 `action_combo` 做 CE 监督。

### Action interface 代码审计（Stage1/2 不监督，Stage3 监督）

2026-08-17 对 `src/nav/v1/models/iw_aligned.py` 做静态审计，当前代码满足：

```text
a_noise_forwarded   = True
combo_logits_head   = True
stage2_no_action_loss = True
stage3_combo_ce     = True
```

对应代码路径：

- `IWActionInterface.noise_tokens()` 将 `a_noise + action_timestep` 编成 shared
  action tokens；
- `forward_core()` 将 action tokens 作为 `shared_action_tokens` 送入 legacy20
  Wan backbone，并 decode 出 `combo_logits`；
- `forward_stage2()` 的 loss 是
  `visual_loss + lambda_pose * pose_loss`，不包含 action CE/flow；
- `forward_stage3_policy()` 在 batch 含 `action_combo` 时使用
  `CE(combo_logits, action_combo)`，并把 continuous action flow 置为 0，仅保留作
  兼容/ablation 输出。

### Combo-schema baseline step200 中间结果

当前运行到 step200：

```text
step100 train visual = 0.1793, eval visual = 0.5723
step180 train visual = 0.2866
step190 train visual = 0.2859
step200 train visual = 0.1555, eval visual = 0.3812
step220 train visual = 0.2065
step230 train visual = 0.2331
step240 train visual = 0.2238
last10 train visual mean around step190 = 0.2662
speed = about 11.1 s/step
```

对照旧曲线：

```text
old StageOne step100 = 0.1680
old StageOne step200 = 0.0852
old cotrain step200 train visual = 0.2322, eval visual = 0.2013
old cotrain 101-200 train visual mean = 0.2672
```

解释：combo-schema baseline 的训练收敛速度目前与旧 cotrain 同量级，
step200 train visual 甚至低于旧 cotrain 同步数；eval 偏高暂不单独判失败，
因为该训练脚本的 eval 不是 fixed-window deterministic eval。最终仍等待
step500/1000 的 last10 train 和 checkpoint 后的固定测评。

2026-08-17 19:48 补充：

```text
current combo 101-200 visual mean = 0.2638
old cotrain   101-200 visual mean = 0.2672
current combo 201-240 visual mean = 0.2119
old cotrain   201-300 visual mean = 0.1988
current combo last10 visual mean  = 0.2386
```

因此新数据/action interface 接旧收敛支路的训练曲线仍在旧 cotrain 同量级内。
step240 pose loss 有一次 spike 到 `0.0408`，但 201–240 pose median 仍为
`0.0029`；旧 cotrain 也存在局部 pose spike，暂不干预，继续看 step300/500
窗口。

2026-08-17 19:59 step300 对齐结果：

| 区间/指标 | current combo-schema baseline | old IW-aligned cotrain | 判断 |
| --- | ---: | ---: | --- |
| 101–200 `train/loss_visual` mean | 0.2638 | 0.2672 | 对齐 |
| 201–300 `train/loss_visual` mean | 0.1970 | 0.1988 | 对齐 |
| step300 `train/loss_visual` | 0.1438 | 0.1574 | current 略好 |
| step300 `eval/loss_visual` | 0.1739 | 0.1717 | 对齐 |
| step300 `train/loss_pose` | 0.00107 | 0.00101 | 对齐 |
| step300 `eval/loss_pose` | 0.00235 | 0.00260 | 对齐 |
| speed | 11.14 s/step | 约 11.39 s/step | 对齐 |

结论：截至 step300，**新数据形式/action combo interface 接到旧收敛
cotrain 支路上已经实证对齐旧版训练曲线**。这并不等于 Formal V1 全部完成；
它只是通过第一关。后续仍等待 step500/1000 checkpoint，然后由 watcher 推进：

1. `fixed_rnull_unified + legacy_iw_move_view`；
2. `fixed_rnull_unified + shared_tail_token`。

2026-08-17 20:05 继续观察：

```text
step310 train visual = 0.2369
step320 train visual = 0.1367
step330 train visual = 0.2054
current combo 301-330 visual mean = 0.1930
current combo last10 visual mean   = 0.1925
old cotrain   301-400 visual mean  = 0.1498
```

301–330 目前略高于旧 cotrain 的完整 301–400 均值，但样本点还少；结合
step300 的强对齐结果和 last10 仍约 0.19，暂不干预，继续等 step400/500。

2026-08-17 20:18 step400 对齐结果：

| 区间/指标 | current combo-schema baseline | old IW-aligned cotrain | 判断 |
| --- | ---: | ---: | --- |
| 201–300 `train/loss_visual` mean | 0.1970 | 0.1988 | 对齐 |
| 301–400 `train/loss_visual` mean | 0.1747 | 0.1498 | current 略高但同量级 |
| step400 `train/loss_visual` | 0.1819 | 0.1523 | current 略高 |
| step400 `eval/loss_visual` | 0.1462 | 0.1404 | 对齐 |
| step400 `train/loss_pose` | 0.00050 | 0.01709 | current 更低 |
| step400 `eval/loss_pose` | 0.00191 | 0.00243 | 对齐 |
| speed | 11.15 s/step | 约 11.39 s/step | 对齐 |

解释：301–400 的 train visual 均值略高于旧 cotrain，但 step400 eval visual
与旧版几乎一致；结合 step300 的完全对齐、速度相同、pose 正常，目前仍判定
combo-schema baseline gate 健康。继续等待 step500 checkpoint 和 step1000
formal watcher。

2026-08-17 20:37 step500 checkpoint 对齐结果：

| 区间/指标 | current combo-schema baseline | old IW-aligned cotrain | 判断 |
| --- | ---: | ---: | --- |
| 301–400 `train/loss_visual` mean | 0.1747 | 0.1498 | current 略高但同量级 |
| 401–500 `train/loss_visual` mean | 0.1227 | 0.2081 | current 更好 |
| step500 `train/loss_visual` | 0.1128 | 0.1088 | 对齐 |
| step500 `eval/loss_visual` | 0.1234 | 0.2381 | current 更好 |
| step500 `train/loss_pose` | 0.00204 | 0.00095 | 同量级 |
| step500 `eval/loss_pose` | 0.00234 | 0.00321 | current 更好 |
| speed | 11.10 s/step | 约 11.39 s/step | 对齐 |

checkpoint：

```text
log/v1_stage2_iw_aligned_re10k_pose_video/
  iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216/
    checkpoints/step_000500.pt
```

只读检查通过：

```text
step = 500
has_model = true
has_optimizer = true
register_mode = legacy_extract_update
current_action_mode = legacy_iw_move_view
combo_action_vocab_size = 144
```

结论：截至 step500，combo-schema baseline 已经稳定达到旧 cotrain 收敛水平；
新 action/data interface 的第一关可视为通过。当前 run 继续到 step1000，
watcher 将在 `step_001000.pt` 落盘后做 fixed-seed eval，再按 gate 进入
`fixed_rnull_unified`。

同一时间，R2R full obs latent 编码 shard1 完成：

```text
output_root = /sharedata/NAV/derived/v1/vln/obs_latents_r2r_full/r2r_standard_20260817_full
manifest    = manifests/encoded_episodes_shard-001-of-002.jsonl
completed   = 6718 / 6718
errors      = 0
encoded_frames = 398362
latent_shape = [16,1,56,112]
```

这说明 Stage3 policy 所需的 full R2R observation latent 数据准备已至少完成该
shard；配合此前 Stage3 preflight，第三阶段变量接口和数据准备条件进一步变实。

2026-08-17 20:45 step500 后继续观察：

```text
step510 train visual = 0.2359, pose = 0.0376
step520 train visual = 1.1351, pose = 0.0021  # visual 单点 spike
step530 train visual = 0.1025, pose = 0.0042
step540 train visual = 0.1180, pose = 0.0016
501–540 visual mean = 0.3979  # 被 step520 单点拉高
521–540 visual mean = 0.1102
```

解释：step520 出现一个 visual spike，但随后 step530/540 立刻回到 0.10–0.12
区间，未表现为发散。旧 cotrain 中也存在单点大 spike（例如 step480、step900）。
因此暂不干预，继续等待 step600/eval 和 step1000 formal gate。

2026-08-17 20:56 step600 对齐结果：

| 区间/指标 | current combo-schema baseline | old IW-aligned cotrain | 判断 |
| --- | ---: | ---: | --- |
| 401–500 `train/loss_visual` mean | 0.1227 | 0.2081 | current 更好 |
| 501–600 `train/loss_visual` mean | 0.2374 | 0.1390 | current 被 step520 spike 拉高 |
| 501–600 `train/loss_visual` median | 0.1102 | 0.1216 | 对齐/current 略好 |
| 501–600 no-spike visual mean | 0.1376 | 0.1390 | 对齐 |
| step600 `train/loss_visual` | 0.0864 | 0.1281 | current 更好 |
| step600 `eval/loss_visual` | 0.0974 | 0.3662 | current 更好 |
| step600 `train/loss_pose` | 0.00077 | 0.00340 | current 更好 |
| step600 `eval/loss_pose` | 0.0249 | 0.0217 | 同量级随机 eval spike |

结论：step520 的单点 spike 没有演化成发散；到 step600，visual train/eval
继续健康，combo-schema baseline 第一关保持通过。继续等待 step1000 checkpoint
和 watcher 的 fixed-seed eval。

2026-08-17 21:03 step600 后继续观察：

```text
step610 train visual = 0.1047, pose = 0.0081
step620 train visual = 1.1883, pose = 0.00058  # 第二个 visual 单点 spike
step630 train visual = 0.1356, pose = 0.00081
step640 train visual = 0.1150, pose = 0.0020
601–640 visual mean = 0.3859  # 被 step620 单点拉高
601–640 no-spike visual mean = 0.1184
621–640 visual mean = 0.1253
```

解释：step620 类似 step520，是单点 visual spike；后续 step630/640 立刻回到
0.11–0.14 区间。结合 step600 eval visual 0.0974，暂不干预，继续等待
step700/eval。

2026-08-17 21:06 step660 继续观察：

```text
step650 train visual = 0.0881, pose = 0.00140
step660 train visual = 0.1249, pose = 0.00105
speed  = 11.09 s/step
memory = 29.26 GiB cuda max
```

判断：step620 后连续两个 log 点恢复到 0.09–0.13 区间，说明 spike 没有形成
持续退化。当前 combo-schema baseline 仍满足“新数据形式/action combo
interface 接到旧收敛 cotrain 支路后，训练速度与收敛速度接近旧版”的第一关。
但这只证明 **interface gate**，还没有证明最终结构 gate；仍需等待
`step_001000.pt` 落盘后的 fixed-seed eval，然后依次跑：

1. `fixed_rnull_unified + legacy_iw_move_view`；
2. `fixed_rnull_unified + shared_tail_token`。

只有这两关也在同一速度/收敛量级通过后，才能说新版正式结构完成对齐。

2026-08-17 21:10 gate watcher 规则修正：

由于当前 RE10K Stage2 训练的 dataloader 会随机采样 window/noise/timestep，
训练曲线存在已观察到的单点 `train/loss_visual` spike；旧版 cotrain 中也有类似
单点 spike。因此原 watcher 只用 `last10 mean` 会在最后 10 个 log 点刚好撞上
spike 时误判失败。现在 watcher 不放宽 loss 阈值，而是同时记录并使用稳健近期
统计：

```text
last10 mean
last10 median
last20 median
last20 no-spike(<0.5) mean
```

通过规则：`last_step >= 1000`，且上述任一近期稳健 visual 指标低于同一阈值，
同时最后一次 eval visual 低于 disaster-check 阈值。这样仍会拒绝持续高 loss
的 run，但不会因为单个随机异常点误杀已经恢复的曲线。

已重启 watcher：

```text
tmux session = nav_formal_v1_gates_after_combo
script       = scripts/watch_combo_schema_then_formal_v1_gates.sh
syntax check = bash -n OK
```

同日代码侧验收审计：

```text
src/nav/v1/models/iw_aligned.py
```

- `forward_core()` 将 `history_latents` 与 `a_hist_combo` 输入
  `roll_register()`；历史动作是 register 更新的 context，而不是独立 move/view
  双流。
- `a_cur_combo` 在当前 baseline 中被 split 成 IW native `move/view`，用于隔离验证
  新 combo schema 是否破坏旧收敛支路；后续 gate 会切到
  `current_action_mode=shared_tail_token`。
- `a_noise + action_timestep` 经 `IWActionInterface.noise_tokens()` 进入
  `shared_action_tokens`，送入 Wan backbone；backbone 返回
  `shared_action_hidden` 后由 `decode()` 产生 `combo_logits`、`action_velocity` 等。
- `forward_stage2()` 只计算 `visual_loss + lambda_pose * pose_loss`，没有 action CE
  或 action flow loss；因此 Stage One/Two 中 action noise/output token 存在但不监督。
- `forward_stage3_policy()` 在 batch 含 `action_combo` 时计算
  `CE(combo_logits, action_combo)`，continuous action flow head 置 0，仅保留兼容/消融。

结论：当前代码满足“兼容最终第三阶段变量；前两阶段 action noise 和输出存在但不
监督”的接口要求。剩余未完成项仍是最终结构收敛 gate：
`fixed_rnull_unified + shared_tail_token`。

2026-08-17 21:12 baseline 继续运行：

```text
latest step     = 690
train visual    = 0.0678
train pose      = 0.00042
last10 mean     = 0.2209  # 被 step620 单点 spike 拉高
last10 median   = 0.1200
last20 median   = 0.1165
last20 no-spike = 0.1270
speed           = 11.11 s/step
step1000 ckpt   = not yet
```

判断：近期稳健统计继续处于旧版 cotrain 同量级；baseline 仍在正常推进。
watcher 当前只等待 `step_001000.pt`，没有进入 fixed-rnull gate。

2026-08-17 21:14 step700/eval700：

```text
step700 train visual = 0.1486
step700 train pose   = 0.00099
step700 eval visual  = 0.1273
step700 eval pose    = 0.00712
speed                = 11.10 s/step
ETA to step1000      = ~55.5 min
```

判断：step700 eval visual 仍在 0.13 左右，baseline 后半段没有出现持续退化；
继续等待 step1000 checkpoint 和 formal watcher 自动进入 fixed-rnull gate。

2026-08-17 21:15 watcher 增加 OOM 防护：

`step_001000.pt` 可能在训练脚本完全退出前先落盘；如果 watcher 只等 checkpoint
就立刻启动 fixed-seed eval，有概率和仍占用 GPU 的训练进程冲突，造成 OOM 假失败。
因此已给 `scripts/watch_combo_schema_then_formal_v1_gates.sh` 增加：

```text
wait_for_checkpoint(...)
wait_for_training_process_exit(...)
run_fixed_eval(...)
```

即 baseline checkpoint 落盘后，还要确认对应 `--run-name` 的
`train_v1_stage2_iw_aligned_re10k_pose_video.py` 进程已经退出，再启动 eval 和后续
gate。`bash -n` 检查通过，watcher 已重启。

2026-08-17 21:18 step720：

```text
step720 train visual = 0.0985
step720 train pose   = 0.00409
last10 mean          = 0.1151
last10 median        = 0.1200
last20 median        = 0.1099
last20 no-spike      = 0.1188
speed                = 11.09 s/step
ETA to step1000      = ~51.7 min
```

判断：step700 之后没有再出现 spike，最近 10 个 log 点均值已回到 0.12 以下；
combo-schema baseline 后半段继续稳定。formal watcher 仍在等待 step1000。

2026-08-17 21:21 step740：

```text
step740 train visual = 0.0854
step740 train pose   = 0.00151
last10 mean          = 0.1100
last10 median        = 0.1066
last20 median        = 0.1097
last20 no-spike      = 0.1178
speed                = 11.14 s/step
ETA to step1000      = ~48.3 min
```

判断：baseline 的新 combo/action interface gate 基本无悬念通过，仍需等
step1000 + fixed-seed eval 形成正式 gate 产物，再启动 fixed-rnull / shared-tail
结构 gate。

2026-08-17 21:25 step760：

```text
step760 train visual = 0.3793  # 中等单点 spike
step760 train pose   = 0.00696
last10 mean          = 0.1433
last10 median        = 0.1202
last20 median        = 0.1200
last20 no-spike      = 0.1372
speed                = 11.09 s/step
ETA to step1000      = ~44.4 min
```

判断：step760 是一个中等 visual spike，但 robust 近期统计仍在 0.12–0.14
区间，且历史上 step520/620 的 spike 都能恢复；暂不干预，继续等待后续 log/eval。

2026-08-17 21:27 step770：

```text
step770 train visual = 0.0951
step770 train pose   = 0.00235
last10 mean          = 0.1356
last10 median        = 0.1066
last20 median        = 0.1149
last20 no-spike      = 0.1315
ETA to step1000      = ~42.5 min
```

判断：step760 后立即恢复到 0.10 以下，确认是单点 spike 而非持续退化。
下一关键观测点是 step800 eval。

2026-08-17 21:36 step800/eval800 与 step810：

```text
step800 train visual = 0.7634  # 较大单点 spike
step800 train pose   = 0.00218
step800 eval visual  = 0.1081
step800 eval pose    = 0.00318
step810 train visual = 0.0894
step810 train pose   = 0.00461
last10 mean          = 0.1983  # 被 step800 spike 拉高
last10 median        = 0.1066
last20 median        = 0.1157
last20 no-spike      = 0.1262
```

判断：step800 train visual 出现较大单点 spike，但同一步 eval visual 反而优于
step700（0.1081 vs 0.1273），且 step810 train 立即恢复到 0.09 左右；因此继续
判定为随机 train window/noise/timestep 的单点异常，不是整体退化。baseline
interface gate 继续健康。

2026-08-17 21:37 step820：

```text
step820 train visual = 0.1299
step820 train pose   = 0.00071
last10 mean          = 0.2014  # 仍被 step800 spike 拉高
last10 median        = 0.1156
last20 median        = 0.1157
last20 no-spike      = 0.1264
eval800 visual       = 0.1081
```

判断：step800 后 step810/820 已恢复到 0.09–0.13；结合 eval800，baseline
interface gate 继续健康。formal watcher 仍在等待 step1000。

2026-08-17 21:44 step850/860：

```text
step850 train visual = 0.4779  # 单点 spike
step850 train pose   = 0.00176
step860 train visual = 0.0924
step860 train pose   = 0.00290
last10 mean          = 0.2101  # 被 step800/850 spike 拉高
last10 median        = 0.1057
last20 median        = 0.1156
last20 no-spike      = 0.1458
eval800 visual       = 0.1081
ETA to step1000      = ~25.9 min
```

判断：step850 后 step860 立即恢复，仍是单点 train sample spike；baseline
interface gate 继续健康。下一关键观测点是 step900/eval900。

2026-08-17 21:52 step880/900：

```text
step880 train visual = 0.0608
step880 train pose   = 0.00083
step900 train visual = 0.0752
step900 train pose   = 0.00136
step900 eval visual  = 0.0768
step900 eval pose    = 0.01980
last10 mean          = 0.1331
last10 median        = 0.0851
last20 median        = 0.0938
last20 no-spike      = 0.1334
ETA to step1000      = ~18.5 min
```

判断：step900 train/eval visual 都进入 0.08 左右，baseline 新数据/action
interface gate 已经基本确定通过。剩余动作是等待 step1000 checkpoint 落盘，
由 watcher 做 fixed-seed eval 和 formal summary，然后自动进入 fixed-rnull gate。

2026-08-17 21:59 step940：

```text
step940 train visual = 0.1168
step940 train pose   = 0.00186
eval900 visual       = 0.0768
last10 mean          = 0.1182
last10 median        = 0.0785
last20 median        = 0.0915
last20 no-spike      = 0.1313
ETA to step1000      = ~11.1 min
```

判断：baseline interface gate 继续稳定，下一步等待 `step_001000.pt` 落盘；
watcher 会先等训练进程退出，再做 fixed-seed eval 和 formal summary。

2026-08-17 22:06 step970：

```text
step970 train visual = 0.0735
step970 train pose   = 0.00090
eval900 visual       = 0.0768
last10 mean          = 0.0783
last10 median        = 0.0757
last20 median        = 0.0829
last20 no-spike      = 0.1098
ETA to step1000      = ~5.5 min
```

判断：baseline 新数据/action interface gate 已达到旧版 cotrain 的低 loss 区间；
继续等待 step1000 checkpoint 和 watcher 自动进入 fixed-seed eval。

2026-08-17 22:13 baseline 1k formal gate 通过：

```text
run = iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216
step1000 train visual = 0.1340
step1000 train pose   = 0.00571
step1000 eval visual  = 0.0779
step1000 eval pose    = 0.00249
last10 mean           = 0.0921
last10 median         = 0.0856
last20 mean           = 0.1126
last20 median         = 0.0856
last20 no-spike       = 0.1126
speed window          = 11.07 s/step
checkpoint            = checkpoints/step_001000.pt
formal gate passed    = true
```

fixed-seed eval：

```text
output_dir = result/v1_stage2_iw_aligned_re10k_pose_video/
  combo_schema_legacybranch_baseline_step1000_fixed_eval

eval/loss_visual_velocity_mse = 0.1426
eval/latent_x0_mse            = 0.0641
eval/latent_x0_rmse           = 0.2375
eval/latent_x0_cosine         = 0.9472
eval/pose_translation_mae     = 0.0608
eval/pose_rotation_angle_deg  = 9.8990
```

formal summary：

```text
result/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gates/
  iw_aligned_combo_schema_legacybranch_re10kfull_1k_20260817_190216/
    baseline_gate_summary.json
```

结论：新 combo/action 数据接口接在旧版可收敛 cotrain 生成支路上，已正式满足
“训练速度、收敛速度和旧版接近；兼容 Stage3 变量；Stage1/2 action noise/output
存在但不监督”的 interface gate。watcher 已自动进入下一关：

```text
run = formal_v1_fixed_rnull_combo_1k_20260817_211521
register_mode = fixed_rnull_unified
current_action_mode = legacy_iw_move_view
```

这一关用于验证把 register 初始/更新切到正式 `fixed_rnull_unified` 后，是否仍能
保持同量级收敛；通过后才会进入 `shared_tail_token`。

2026-08-17 22:15 fixed-rnull gate 启动正常：

```text
run = formal_v1_fixed_rnull_combo_1k_20260817_211521
register_mode        = fixed_rnull_unified
current_action_mode  = legacy_iw_move_view
step1 train visual   = 1.2036
step10 train visual  = 0.7467
step10 train pose    = 0.2832
speed window         = 9.53 s/step
cuda max memory      = 29.26 GiB
```

判断：fixed-rnull 正式结构第一关已正常启动，显存/速度和 baseline 同量级；
step10 仍属于 warmup 初期，不能据此判定收敛对齐。下一关键点是 step50/eval50。

2026-08-17 22:22 fixed-rnull step50/eval50：

```text
run = formal_v1_fixed_rnull_combo_1k_20260817_211521
step50 train visual = 0.3899
step50 train pose   = 0.01117
step50 eval visual  = 0.4657
step50 eval pose    = 0.03503
recent train rows   = [1,10,20,30,40,50]
recent visual mean  = 0.6645
recent visual median= 0.5926
speed window        = 11.03 s/step
cuda max memory     = 29.26 GiB
```

判断：fixed-rnull 已正常下降，但 step50/eval50 仍明显高于已收敛 baseline；
这仍属早期 warmup，不能直接判失败。下一关键点是 step100/eval100：如果 visual
不能快速追到 0.2–0.3 量级，需要怀疑 `fixed_rnull_unified` 本身造成收敛速度
损失；如果继续快速下降，则让它跑满 1k 做正式 gate。

2026-08-17 22:32 fixed-rnull step100/eval100：

```text
step100 train visual = 0.9805  # train 单点 spike，不能单独作为趋势
step100 train pose   = 0.03516
step100 eval visual  = 0.3349
step100 eval pose    = 0.02313
recent rows          = [10,20,30,40,50,60,70,80,90,100]
recent visual mean   = 0.5905
recent visual median = 0.5181
recent no-spike mean = 0.3955
speed window         = 11.02 s/step
```

判断：eval visual 从 step50 的 0.4657 降到 step100 的 0.3349，有实质改善；
但仍明显慢于 baseline interface gate。当前不能判失败，因为 train step100 是
spike，且 eval 仍在下降。需要继续观察 step150/200：如果 eval 无法进入 0.2
附近，则 `fixed_rnull_unified` 大概率是正式结构收敛变慢的主要来源。

2026-08-17 22:41 fixed-rnull step150/eval150：

```text
step150 train visual = 0.5641  # train 点仍有 spike/高方差
step150 train pose   = 0.00152
step150 eval visual  = 0.2623
step150 eval pose    = 0.00243
recent rows          = [60,70,80,90,100,110,120,130,140,150]
recent visual mean   = 0.4777
recent visual median = 0.3521
recent no-spike mean = 0.3410
speed window         = 11.01 s/step
```

判断：eval visual 继续从 0.3349 降到 0.2623，说明 `fixed_rnull_unified` 并未
破坏学习链路；但相比 baseline interface gate，收敛显著更慢。下一关键点是
step200/eval200：若能接近 0.2，则继续跑满 1k；若停在 0.25 以上，需要考虑
fixed-rnull 作为正式结构的收敛代价是否可接受。

2026-08-17 22:51 fixed-rnull step200/eval200：

```text
step200 train visual = 0.2820
step200 train pose   = 0.00211
step200 eval visual  = 0.2305
step200 eval pose    = 0.00246
recent rows          = [110,120,130,140,150,160,170,180,190,200]
recent visual mean   = 0.2816
recent visual median = 0.2388
recent no-spike mean = 0.3021
speed window         = 11.03 s/step
```

判断：fixed-rnull eval visual 从 0.4657 → 0.3349 → 0.2623 → 0.2305 持续下降，
说明正式 Rnull/register 结构可学习；但相比 baseline interface gate 明显慢。当前
不停止，继续跑到至少 step300/eval300；如果 step300 进入 0.18–0.20 区间，则
有希望在 1k 达到 formal threshold，否则需要把“fixed-rnull 收敛代价”作为结构
问题记录。

2026-08-17 22:52 fixed-rnull 与 baseline/old cotrain 早期区间对比：

| 区间/指标 | baseline combo | fixed-rnull | old cotrain |
| --- | ---: | ---: | ---: |
| 001–050 train visual mean | 0.5881 | 0.6645 | 0.6835 |
| 051–100 train visual mean | 0.3380 | 0.6244 | 0.3629 |
| 101–150 train visual mean | 0.2790 | 0.3310 | 0.3151 |
| 151–200 train visual mean | 0.2486 | 0.2322 | 0.2194 |
| eval100 visual | 0.5723 | 0.3349 | 0.3130 |
| eval200 visual | 0.3812 | 0.2305 | 0.2013 |
| speed | ~11.1 s/step | ~11.0 s/step | ~11.4 s/step |

补充判断：从 train 区间看，fixed-rnull 在 151–200 已经追到 baseline/old
同量级；从 eval 看，它比 old cotrain 慢约 0.03，但比本次 baseline combo 的
eval200 更好。当前更准确的判断是：fixed-rnull 前 100 step 慢热，但 200 step
时不能判结构失败。继续观察 step300/eval300。

2026-08-17 23:03 fixed-rnull step250/eval250 与 step260：

```text
step250 eval visual  = 0.2219
step250 eval pose    = 0.02212
step260 train visual = 0.1353
step260 train pose   = 0.00276
recent rows          = [170,180,190,200,210,220,230,240,250,260]
recent visual mean   = 0.2155
recent visual median = 0.2215
recent no-spike mean = 0.2434
```

判断：eval250 相比 eval200 只小幅下降（0.2305 → 0.2219），但 step260 train
visual 已到 0.135，说明 fixed-rnull 有 batch-level 好样本且训练未坏。仍需
step300/eval300 再判断是否能进入 0.18–0.20 量级。

2026-08-17 23:10 fixed-rnull step300/eval300：

```text
step300 train visual = 0.2360
step300 train pose   = 0.00172
step300 eval visual  = 0.3051
step300 eval pose    = 0.00204
recent rows          = [210,220,230,240,250,260,270,280,290,300]
recent visual mean   = 0.2241
recent visual median = 0.2407
recent no-spike mean = 0.2364
```

判断：step300 train 仍在 0.22–0.24 量级，但 eval300 从 eval250 的 0.2219
反弹到 0.3051，没有达到预期的 0.18–0.20 区间。当前不能宣称 fixed-rnull 的
收敛速度与旧版接近；更准确地说，它已经证明“可学习”，但目前显示出明显
收敛代价。由于 eval 随 random window/noise/timestep 有波动，继续观察
step350/400 后再决定是否让它跑满 1k 作为 formal gate。

同日代码核查：

```text
src/nav/v1/models/iw_aligned.py::SpatialRegisterMemory
```

确认 `fixed_rnull_unified` 实现符合当前正式设定：

- `_fixed_rnull()` 返回非 learnable 的全零 `[B,16,4,H,W]` Register；
- `extract()` 在 fixed-rnull 模式下不会使用 adaptive-pooled first chunk 作为
  initial register，而是 `R_null + unified_blocks(first_chunk, A_hist)`；
- `update()` 对后续 history chunks 继续使用同一组 `unified_blocks`；
- `forward_core()` 中 `roll_register(history, a_hist_combo)` 确实会遍历所有
  history chunks；没有少用 history。

因此 step300 的变慢不是明显实现错误，更可能来自“从零 Rnull 写入全部历史”的
结构难度。

2026-08-17 23:20 fixed-rnull step350/eval350：

```text
step350 train visual = 0.1471
step350 train pose   = 0.00403
step350 eval visual  = 0.1566
step350 eval pose    = 0.00251
recent rows          = [260,270,280,290,300,310,320,330,340,350]
recent visual mean   = 0.2081
recent visual median = 0.1864
recent no-spike mean = 0.2173
speed window         = 11.05 s/step
```

判断：eval350 从 eval300 的 0.3051 显著回落到 0.1566，已经进入预期的
0.18–0.20 以下区间；因此 eval300 更像随机 eval window/noise spike，而非结构性
失败。当前 fixed-rnull 更准确的状态是：前 100–200 step 明显慢热，但 350 step
已经追到可接受区间，有希望在 1k formal gate 达到通过阈值。继续跑满 1k。

2026-08-17 23:30 fixed-rnull step400/eval400 与 step410：

```text
step400 eval visual  = 0.2071
step400 eval pose    = 0.00164
step410 train visual = 0.1174
step410 train pose   = 0.00203
recent rows          = [320,330,340,350,360,370,380,390,400,410]
recent visual mean   = 0.1703
recent visual median = 0.1580
recent no-spike mean = 0.1947
speed window         = 12.06 s/step
```

判断：eval400 相比 eval350 有回弹（0.1566 → 0.2071），但 step410 train
visual 已到 0.117，且 recent median 为 0.158，说明训练侧已经接近 formal
threshold。当前不应提前停止；fixed-rnull 的 eval 波动较大，但整体仍有希望在
1k 通过 robust train gate。继续跑满 1k，由 watcher 统一做 fixed-seed eval 和
formal summary。

2026-08-17 23:34 fixed-rnull step430 更新：

```text
step430 train visual = 0.1890
step430 train pose   = 0.00214
step400 eval visual  = 0.2071
step400 eval pose    = 0.00164
recent rows          = [340,350,360,370,380,390,400,410,420,430]
recent visual mean   = 0.1698
recent visual median = 0.1683
speed window         = 11.02 s/step
```

判断：step430 出现一个较难训练窗口，使 last10 mean/median 暂时回到
0.17 左右。结合 step350/eval350 与 step410 train visual，当前证据仍支持
“fixed-rnull 可学但波动更大、warmup 更慢”，不能提前宣称与旧版完全对齐；
继续等待 step500/1000，由 formal watcher 负责最终 gate。

2026-08-17 23:37 fixed-rnull step440/450 更新：

```text
step440 train visual = 0.1130
step450 train visual = 0.1055
step450 train pose   = 0.00064
step450 eval visual  = 0.1662
step450 eval pose    = 0.02008
last10 train visual mean   = 0.1496
last10 train visual median = 0.1365
last20 train visual median = 0.1580
speed window               = 11.00 s/step
```

判断：step440/450 已经把 recent train visual 拉到 formal threshold 以下，
eval450 也回到 0.1662，说明 fixed-rnull 的主要问题更像 warmup 与随机窗口
波动，而不是结构链路错误。当前仍不能结束 gate：正式通过需要 step1000
checkpoint、fixed-seed eval、以及后续 shared-tail gate。

2026-08-17 23:47 fixed-rnull step500/eval500：

```text
step500 train visual = 0.0819
step500 train pose   = 0.00102
step500 eval visual  = 0.1573
step500 eval pose    = 0.00244
last10 train visual mean   = 0.1240
last10 train visual median = 0.1152
last20 train visual mean   = 0.1517
last20 train visual median = 0.1513
speed window               = 11.03 s/step
```

同口径中点对比：

| run | step500 train visual | step500 eval visual | train 410–500 mean | train 410–500 median |
| --- | ---: | ---: | ---: | ---: |
| baseline combo / legacy branch | 0.1128 | 0.1234 | 0.1227 | 0.1184 |
| fixed-rnull unified | 0.0819 | 0.1573 | 0.1240 | 0.1152 |
| old twopass cotrain | 0.1088 | 0.2381 | 0.2081 | 0.1499 |

判断：到 step500，fixed-rnull 的训练侧已经与 baseline combo 基本同量级，
甚至 `410–500 median` 略低；eval500 仍高于 baseline eval500，但处在可接受
范围，且低于 old cotrain 的随机 eval500。当前可以说 fixed-rnull 已经通过
“中点趋势检查”，但 formal pass 仍需 step1000 checkpoint 与 fixed-seed eval。

2026-08-17 23:56 fixed-rnull step550/eval550：

```text
step550 train visual = 0.0875
step550 train pose   = 0.00885
step550 eval visual  = 0.1185
step550 eval pose    = 0.01130
last10 train visual mean   = 0.1102
last10 train visual median = 0.0925
last20 train visual mean   = 0.1299
last20 train visual median = 0.1152
speed window               = 11.03 s/step
```

判断：eval550 从 eval500 的 0.1573 继续降到 0.1185，训练 recent stats 也
稳定低于 threshold。fixed-rnull 现在已经基本证明“训练速度/收敛速度可接近
旧版”，剩余是形式验收：跑满 1k、生成 checkpoint、fixed-seed eval，并进入
shared-tail gate。

2026-08-18 00:05 fixed-rnull step600/eval600：

```text
step600 train visual = 0.1033
step600 train pose   = 0.00623
step600 eval visual  = 0.1069
step600 eval pose    = 0.00205
last10 train visual mean   = 0.1107
last10 train visual median = 0.1025
last20 train visual median = 0.1120
speed window               = 11.00 s/step
```

判断：eval600 继续下降到 0.1069，fixed-rnull 在 500–600 区间已经形成稳定
收敛段。当前证据足以支持 fixed-rnull 结构“不是收敛瓶颈”；但 formal V1
目标仍未完成，因为还缺 step1000 checkpoint/fixed-seed eval 和 shared-tail
最终 gate。

2026-08-18 00:15 fixed-rnull step650/eval650：

```text
step650 train visual = 0.1008
step650 train pose   = 0.00130
step650 eval visual  = 0.1425
step650 eval pose    = 0.00302
last10 train visual mean   = 0.1157
last10 train visual median = 0.1078
last20 train visual mean   = 0.1130
last20 train visual median = 0.1025
speed window               = 11.03 s/step
```

判断：eval650 相比 eval600 回弹（0.1069 → 0.1425），但仍在稳定区间；
训练 recent mean/median 均远低于 candidate threshold 0.18。fixed-rnull gate
目前继续保持通过趋势。预计还需约 60–65 分钟到 step1000。

2026-08-18 00:24 fixed-rnull step700/eval700：

```text
step700 train visual = 0.1011
step700 train pose   = 0.00101
step700 eval visual  = 0.1240
step700 eval pose    = 0.00340
last10 train visual mean   = 0.1118
last10 train visual median = 0.1106
last20 train visual mean   = 0.1103
last20 train visual median = 0.1061
speed window               = 11.02 s/step
```

判断：step700/eval700 继续稳定，fixed-rnull 的 600–700 段已经呈现低方差
收敛状态。除非后 300 step 出现异常，fixed-rnull gate 大概率会通过；真正
未验证的部分已经转移到 `shared_tail_token` 最终结构。

2026-08-18 00:33 fixed-rnull step750/eval750：

```text
step740 train visual = 0.2163  # random-window spike
step750 train visual = 0.0811
step750 train pose   = 0.00084
step750 eval visual  = 0.0905
step750 eval pose    = 0.01877
last10 train visual mean   = 0.1152
last10 train visual median = 0.1062
last20 train visual mean   = 0.1158
last20 train visual median = 0.1062
speed window               = 11.02 s/step
```

判断：step740 出现单点 spike，但 step750/eval750 明显回落，eval visual
达到 0.0905。fixed-rnull gate 的通过趋势已经很强；剩余重点是等 step1000
checkpoint、fixed-seed eval、以及后续 `shared_tail_token` gate。

2026-08-18 00:43 fixed-rnull step800/eval800：

```text
step800 train visual = 0.0791
step800 train pose   = 0.00270
step800 eval visual  = 0.1098
step800 eval pose    = 0.00453
last10 train visual mean   = 0.1007
last10 train visual median = 0.0870
last20 train visual mean   = 0.1075
last20 train visual median = 0.1022
speed window               = 11.04 s/step
```

判断：800 step 仍维持低 loss 与稳定速度。fixed-rnull 已经在 500–800 区间
持续满足 candidate gate 的 recent statistics；下一步只需等待 1000 step 的
formal summary 与 fixed-seed eval，再进入 shared-tail。

2026-08-18 00:52 fixed-rnull step850/eval850：

```text
step840 train visual = 0.1537
step850 train visual = 0.0740
step850 train pose   = 0.00316
step850 eval visual  = 0.1123
step850 eval pose    = 0.00119
last10 train visual mean   = 0.0915
last10 train visual median = 0.0890
last20 train visual mean   = 0.1047
last20 train visual median = 0.0955
speed window               = 11.02 s/step
```

判断：fixed-rnull 后段继续稳定，recent train visual 已压到约 0.09–0.10
量级。当前 fixed-rnull gate 基本只剩 step1000 checkpoint 与 watcher 产物
验证；shared-tail 仍未启动。

2026-08-18 01:01 fixed-rnull step900/eval900：

```text
step900 train visual = 0.0931
step900 train pose   = 0.00132
step900 eval visual  = 0.3124  # random-window eval spike
step900 eval pose    = 0.00708
last10 train visual mean   = 0.1091
last10 train visual median = 0.0927
last20 train visual mean   = 0.1049
last20 train visual median = 0.0919
speed window               = 11.01 s/step
```

判断：eval900 相比 eval850 明显回弹，但 train recent statistics 仍稳定在
0.09–0.11 区间；按照 watcher 的正式规则，eval 只是 disaster check
（candidate 阈值 0.6），不作为主要收敛判据。当前 fixed-rnull gate 仍保持
通过趋势，等待 step1000 checkpoint 和 fixed-seed eval。

2026-08-18 01:11 fixed-rnull step950/eval950：

```text
step950 train visual = 0.0770
step950 train pose   = 0.00043
step950 eval visual  = 0.0776
step950 eval pose    = 0.01823
last10 train visual mean   = 0.0999
last10 train visual median = 0.0827
last20 train visual mean   = 0.0959
last20 train visual median = 0.0859
speed window               = 10.95 s/step
```

判断：eval950 从 eval900 的随机 spike 回落到 0.0776，fixed-rnull gate
几乎确定会通过 robust recent statistics。剩余步骤：step1000 checkpoint、
fixed-seed eval、register gate report、fixed-rnull summary，然后启动
`shared_tail_token`。

2026-08-18 01:23 fixed-rnull formal gate 通过：

```text
run      = formal_v1_fixed_rnull_combo_1k_20260817_211521
ckpt     = log/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_fixed_rnull_combo_1k_20260817_211521/checkpoints/step_001000.pt
ckpt size = 8.56 GB

step1000 train visual = 0.0990
step1000 train pose   = 0.00412
step1000 eval visual  = 0.0912
step1000 eval pose    = 0.00230

last10 train visual mean          = 0.1398
last10 train visual median        = 0.0900
last20 train visual mean          = 0.1245
last20 train visual median        = 0.0927
last20 train visual no-spike mean = 0.1024
speed window                      = 11.04 s/step
summary passed                    = true
```

formal 产物：

```text
result/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gates/
  formal_v1_fixed_rnull_combo_1k_20260817_211521/
    fixed_rnull_gate_summary.json
    register_gate_report.json

result/v1_stage2_iw_aligned_re10k_pose_video/
  fixed_rnull_unified_combo_schema_step1000_fixed_eval/eval_metrics.json
```

fixed-seed eval：

```text
eval/loss_visual_velocity_mse = 0.1525
eval/latent_x0_mse           = 0.0722
eval/latent_x0_cosine        = 0.9408
eval/pose_translation_mae    = 0.0334
eval/pose_rotation_angle_deg = 6.0073
```

与 baseline combo 的 paired comparison：

```text
mean visual diff candidate - baseline = -0.00736
candidate last10 visual mean          = 0.1398
baseline  last10 visual mean          = 0.0921
candidate last eval visual            = 0.0912
baseline  last eval visual            = 0.0779
```

判断：`fixed_rnull_unified + legacy_iw_move_view` 已正式通过。它前期 warmup
更慢，末尾有 step960 spike，但 robust recent stats、last eval、fixed-seed eval
均支持它和旧版收敛速度/训练速度接近。该结论只覆盖 Register 结构切换；
最终 shared-tail 仍需单独 gate。

同日 shared-tail 首次启动失败与修复：

```text
failed run = formal_v1_fixed_rnull_shared_tail_1k_20260817_211521
error      = RuntimeError: tensor a (56) must match tensor b (112) at dim 4
location   = Infinite-World-legacy20/infworld/models/dit_model.py
cause      = disable_native_action_embedding=True 时，零 action embedding 被构造成
             [B, dim, T, H, W]，但此时已经在 patch_embedding 后，应为可广播
             的 [B, dim, T, 1, 1]。
fix        = 将 disabled native action embedding 的空间维改为 1x1。
```

已新增恢复脚本：

```bash
bash scripts/resume_formal_v1_shared_tail_gate.sh
```

该脚本只恢复最终 shared-tail gate，不重跑已通过的 baseline / fixed-rnull：

1. 使用 `formal_v1_fixed_rnull_combo_1k_20260817_211521` 作为 baseline；
2. 启动 `fixed_rnull_unified + shared_tail_token` 的 1000-step 正式训练；
3. 运行 fixed-seed eval；
4. 生成 `shared_tail_gate_report.json` 和 `shared_tail_gate_summary.json`。

注意：2026-08-18 01:24 起 Codex 当前 shell 处于受限 sandbox，`torch.cuda.is_available=False`，
无法直接从该 shell 重启 GPU 训练；需要在 CUDA 可见的宿主/IDE 终端执行上述恢复脚本，
或等待后续具备 GPU 访问的执行环境继续。

2026-08-18 01:26 状态复查：

```text
fixed-rnull formal gate:
  run     = formal_v1_fixed_rnull_combo_1k_20260817_211521
  status  = passed
  ckpt    = checkpoints/step_001000.pt

shared-tail first failed run:
  run        = formal_v1_fixed_rnull_shared_tail_1k_20260817_211521
  train rows = 0
  reason     = pre-step shape mismatch before patch

shared-tail resumed run:
  status = not started yet
  resume logs = none

current Codex shell:
  torch.cuda.is_available = False
  device_count            = 0
```

代码状态：

```text
Infinite-World-legacy20/infworld/models/dit_model.py
  disable_native_action_embedding branch 已修成 [B, dim, T, 1, 1]

scripts/resume_formal_v1_shared_tail_gate.sh
  exists      = true
  executable  = true
  bash -n     = pass
```

下一步仍是从 CUDA 可见终端运行：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
bash scripts/resume_formal_v1_shared_tail_gate.sh
```

2026-08-18 01:27 再次复查：

```text
shared-tail resumed run = not found
resume logs             = none
failed first run        = formal_v1_fixed_rnull_shared_tail_1k_20260817_211521
failed first run rows   = train 0 / eval 0 / checkpoint 0

current Codex shell:
  torch.cuda.is_available = False
  device_count            = 0
```

结论：最终 shared-tail gate 仍未启动。当前代码修复和恢复脚本均已准备好，
但该 Codex sandbox 无 CUDA/NVML 访问，也无法连接宿主 tmux socket，因此无法
直接启动正式 GPU 训练。继续推进需要外部 CUDA 可见终端执行恢复脚本。

2026-08-18 01:31 环境恢复并重启 shared-tail formal gate：

复查发现 CUDA/tmux 已恢复正常：

```text
torch.cuda.is_available = True
device_count            = 2
GPU0                    = RTX 6000 Ada, nearly free
tmux                    = available
```

已启动新的 tmux session：

```text
tmux session = nav_formal_v1_shared_tail_fixedpatch
run          = formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124
resume log   = log/v1_stage2_iw_aligned_re10k_pose_video/
               resume_formal_v1_shared_tail_gate_formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124.log
device       = cuda:0
```

启动命令：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV
DEVICE=cuda:0 bash scripts/resume_formal_v1_shared_tail_gate.sh
```

2026-08-18 01:41 shared-tail step50/eval50：

```text
step1   train visual = 0.2183
step10  train visual = 0.4166
step40  train visual = 0.4700
step50  train visual = 0.1556
step50  eval visual  = 0.1666
step50  eval pose    = 0.02342
speed window          = 11.08 s/step
cuda max memory       = 29.26 GB
```

判断：shared-tail 不再 shape crash，说明 `disable_native_action_embedding`
的 1x1 broadcast 修复生效。早期 10–40 step visual loss 明显抖动，但 step50
已回落到 0.1556，eval50 为 0.1666，处于 candidate gate 可接受范围。继续观察
step100/200，不能仅凭 step50 宣称 final gate 通过。

2026-08-18 02:21 shared-tail step260 更新：

```text
tmux session = nav_formal_v1_shared_tail_fixedpatch
run          = formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124
pid          = 4133447
device       = cuda:0
GPU memory   = 32.78 GB process / 29.26 GB cuda_max_memory logged

step100 train visual = 0.3889
step100 eval visual  = 0.1214
step150 train visual = 0.1646
step150 eval visual  = 0.1123
step200 train visual = 0.0624
step200 eval visual  = 0.0724
step250 train visual = 0.0595
step250 eval visual  = 0.0768
step260 train visual = 0.0424
last10 train visual mean   = 0.0703
last10 train visual median = 0.0717
last20 train visual mean   = 0.1274
last20 train visual median = 0.0924
speed window               = 12.12 s/step
```

同同步数对比：

| run | eval50 | eval100 | eval150 | eval200 | eval250 | train 170–260 mean | train 170–260 median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-rnull + legacy move/view | 0.4658 | 0.3349 | 0.2623 | 0.2305 | 0.2219 | 0.2155 | 0.2215 |
| shared-tail fixedpatch | 0.1666 | 0.1214 | 0.1123 | 0.0724 | 0.0768 | 0.0703 | 0.0717 |
| baseline combo legacy branch | n/a | 0.5723 | n/a | 0.3812 | n/a | 0.2248 | 0.2285 |

判断：shared-tail 在 100 step 后迅速进入低 visual loss 区间，step200/250 eval
已低于 0.08，170–260 train mean/median 显著优于 fixed-rnull 和 baseline。
这说明当前最新结构不仅可运行，而且早期收敛性很好。仍需跑满 step1000，
生成 `shared_tail_gate_report.json` / `shared_tail_gate_summary.json` 后才能
正式宣称 final gate 通过。

2026-08-18 02:29 shared-tail step300/eval300：

```text
step290 train visual = 0.1399
step300 train visual = 0.1093
step300 train pose   = 0.00122
step300 eval visual  = 0.1550
step300 eval pose    = 0.00125
last10 train visual mean   = 0.0838
last10 train visual median = 0.0747
last20 train visual mean   = 0.0999
last20 train visual median = 0.0868
speed window               = 11.05 s/step
```

判断：eval300 从 eval250 的 0.0768 回弹到 0.1550，但 train recent 仍在
0.08–0.10 区间，且 pose 很低。暂按随机 eval window/noise 波动处理，不停止；
继续观察 eval350/400 是否回到低位。

2026-08-18 02:38 shared-tail step350/eval350：

```text
step330 train visual = 0.0818
step350 train visual = 0.0570
step350 train pose   = 0.00153
step350 eval visual  = 0.0476
step350 eval pose    = 0.00126
last10 train visual mean   = 0.0861
last10 train visual median = 0.0782
last20 train visual mean   = 0.0798
last20 train visual median = 0.0782
speed window               = 11.09 s/step
```

判断：eval350 从 eval300 的 0.1550 回落到 0.0476，确认 eval300 更像
random-window/noise spike，而不是结构性不收敛。到 350 step，shared-tail
final 结构已经展现出强收敛证据：train recent 低且稳定、eval 低、pose 低。
继续跑到 500/1000 形成 formal summary。

2026-08-18 02:47 shared-tail step400/eval400：

```text
step380 train visual = 0.0261
step400 train visual = 0.0735
step400 train pose   = 0.00174
step400 eval visual  = 0.1175
step400 eval pose    = 0.00156
last10 train visual mean   = 0.0667
last10 train visual median = 0.0658
last20 train visual mean   = 0.0753
last20 train visual median = 0.0710
speed window               = 11.06 s/step
```

判断：train recent 继续保持低位，eval400 从 eval350 的 0.0476 回弹到
0.1175，但仍远低于 disaster threshold，且 pose 很低。当前趋势仍是
“稳定收敛 + eval 随机窗口波动”，没有结构性失败信号。继续等待 step500
和最终 step1000 formal summary。

2026-08-18 03:06 shared-tail step500/eval500 中点检查：

```text
step450 train visual = 0.0399
step450 eval visual  = 0.0879
step480 train visual = 0.0457
step500 train visual = 0.0270
step500 train pose   = 0.00048
step500 eval visual  = 0.0794
step500 eval pose    = 0.00250
last10 train visual mean   = 0.0464
last10 train visual median = 0.0388
last20 train visual mean   = 0.0565
last20 train visual median = 0.0495
last30 train visual mean   = 0.0656
last30 train visual median = 0.0576
speed window               = 11.09 s/step
```

同同步数对比：

| run | step500 train visual | step500 eval visual | train 210–500 mean | train 210–500 median | train 410–500 mean | train 410–500 median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline combo / legacy branch | 0.1128 | 0.1234 | 0.1648 | 0.1473 | 0.1227 | 0.1184 |
| fixed-rnull + legacy move/view | 0.0819 | 0.1573 | 0.1758 | 0.1655 | 0.1240 | 0.1152 |
| shared-tail fixedpatch | 0.0270 | 0.0794 | 0.0656 | 0.0576 | 0.0464 | 0.0388 |

判断：shared-tail 在中点 500 step 已明显优于两个参照 run：train recent 更低，
eval500 也更低。当前可较强地判断最新结构具备良好训练收敛性；剩余工作是
继续跑满 1000 step，让恢复脚本自动生成 fixed-seed eval、`shared_tail_gate_report.json`
和 `shared_tail_gate_summary.json`，完成 formal 证据。

2026-08-18 03:09 shared-tail 终端/环境复查：

```text
tmux session = nav_formal_v1_shared_tail_fixedpatch alive
train process = alive on cuda:0
latest step   = 520
step520 train visual = 0.0407
step520 train pose   = 0.00100
last10 train visual mean   = 0.0485
last10 train visual median = 0.0432
last20 train visual mean   = 0.0549
last20 train visual median = 0.0483
speed window = 11.07 s/step
checkpoint = not yet, save-every=1000
```

判断：当前终端、tmux、CUDA 训练进程均正常；shared-tail 在 500 step 后继续
低位训练，没有再次出现 shape crash 或日志停滞。继续等待 750/1000 step
和最终 formal gate summary。

2026-08-18 03:15 shared-tail step550/eval550：

```text
step540 train visual = 0.0253
step550 train visual = 0.0376
step550 train pose   = 0.00485
step550 eval visual  = 0.0533
step550 eval pose    = 0.01053
last10 train visual mean   = 0.0419
last10 train visual median = 0.0392
last20 train visual mean   = 0.0491
last20 train visual median = 0.0409
speed window = 11.07 s/step
```

判断：eval550 回到较低水平，和 eval350 的低点一致支持“结构可收敛”的判断。
step500 之后 train recent 没有退化，shared-tail 当前比 baseline/fixed-rnull
同步数值更优的趋势继续成立。仍需跑满 1000 step 取得 checkpoint 和 final gate。

2026-08-18 03:24 shared-tail step600/eval600：

```text
step600 train visual = 0.0529
step600 train pose   = 0.00641
step600 eval visual  = 0.0399
step600 eval pose    = 0.00152
last10 train visual mean   = 0.0507
last10 train visual median = 0.0476
last20 train visual mean   = 0.0485
last20 train visual median = 0.0419
last30 train visual mean   = 0.0546
last30 train visual median = 0.0495
speed window = 11.06 s/step
```

判断：eval600 visual 达到当前最低区间，pose loss 同时低位，说明 shared-tail
并没有牺牲 3D/pose 分支换取 visual 收敛。到 step600 为止，最新结构的
训练收敛性证据已经较强；最终仍以 step1000 checkpoint + fixed-seed eval
+ gate summary 为 formal 结论。

2026-08-18 03:34 shared-tail step650/eval650：

```text
step640 train visual = 0.0363
step650 train visual = 0.0382
step650 train pose   = 0.00093
step650 eval visual  = 0.0602
step650 eval pose    = 0.00233
last10 train visual mean   = 0.0537
last10 train visual median = 0.0522
last20 train visual mean   = 0.0478
last20 train visual median = 0.0419
last30 train visual mean   = 0.0507
last30 train visual median = 0.0442
speed window = 11.09 s/step
```

判断：eval650 相比 eval600 有轻微回弹，但仍在低 visual loss 区间；
train recent 在 600–650 区间保持稳定，pose loss 也没有发散。当前证据继续
支持 shared-tail fixedpatch 结构可训且收敛快于前两个 formal 参照。

2026-08-18 03:43 shared-tail step700/eval700：

```text
step680 train visual = 0.1208  # 单点/窗口波动
step690 train visual = 0.0433
step700 train visual = 0.0387
step700 train pose   = 0.00035
step700 eval visual  = 0.0662
step700 eval pose    = 0.00310
last10 train visual mean   = 0.0530
last10 train visual median = 0.0424
last20 train visual mean   = 0.0519
last20 train visual median = 0.0430
last30 train visual mean   = 0.0500
last30 train visual median = 0.0421
speed window = 11.07 s/step
```

判断：step680 出现单点较高 visual loss，但 step690/700 很快回到低位，
last20/30 仍稳定在约 0.05。eval700 相比 eval600/650 略高，但仍属于低
visual loss 区间，未显示结构性退化。继续等待 750 和最终 1000-step formal gate。

2026-08-18 03:53 shared-tail step750/eval750：

```text
step740 train visual = 0.1049  # 单点/窗口波动
step750 train visual = 0.0254
step750 train pose   = 0.00051
step750 eval visual  = 0.0413
step750 eval pose    = 0.01717
last10 train visual mean   = 0.0582
last10 train visual median = 0.0490
last20 train visual mean   = 0.0560
last20 train visual median = 0.0522
last30 train visual mean   = 0.0513
last30 train visual median = 0.0430
speed window = 11.07 s/step
```

判断：step750/eval750 是 shared-tail 的第二个强验证点：eval visual 从
eval700 的 0.0662 回到 0.0413，接近 eval600 的 0.0399。train recent
受 step680/740 两个单点高值影响均值略高，但 median 仍稳定在低位。当前可以认为
latest shared-tail 结构已通过 750-step 中程收敛观察；最终 completion audit
仍等待 step1000 checkpoint、fixed-seed eval 和 shared-tail gate summary。

2026-08-18 04:02 shared-tail step800/eval800：

```text
step760 train visual = 0.0243
step770 train visual = 0.0342
step780 train visual = 0.0394
step790 train visual = 0.0646
step800 train visual = 0.0331
step800 train pose   = 0.00349
step800 eval visual  = 0.0634
step800 eval pose    = 0.00580
last10 train visual mean   = 0.0475
last10 train visual median = 0.0391
last20 train visual mean   = 0.0502
last20 train visual median = 0.0404
last30 train visual mean   = 0.0504
last30 train visual median = 0.0421
last50 train visual mean   = 0.0529
last50 train visual median = 0.0467
speed window = 11.09 s/step
```

判断：750–800 区间 train loss 稳定，eval800 从 eval750 低点回弹到 0.0634，
但仍处于 550–800 后半程低位波动区间。shared-tail fixedpatch 到 800 step
没有发现退化或训练失败迹象。

2026-08-18 04:12 shared-tail step850/eval850：

```text
step810 train visual = 0.0419
step820 train visual = 0.0316
step830 train visual = 0.0294
step840 train visual = 0.1117  # 单点/窗口波动
step850 train visual = 0.0412
step850 train pose   = 0.00069
step850 eval visual  = 0.0505
step850 eval pose    = 0.00107
last10 train visual mean   = 0.0451
last10 train visual median = 0.0368
last20 train visual mean   = 0.0517
last20 train visual median = 0.0413
last30 train visual mean   = 0.0524
last30 train visual median = 0.0423
last50 train visual mean   = 0.0511
last50 train visual median = 0.0417
speed window = 11.06 s/step
```

判断：eval850 visual = 0.0505，pose = 0.00107；750–850 后半程保持
稳定低位。尽管偶有单点高值，median 与 eval 均支持这不是结构性退化。
当前 shared-tail fixedpatch 已完成 85% 训练，等待 900/950/1000 final gate。

2026-08-18 04:21 shared-tail step900/eval900：

```text
step860 train visual = 0.0302
step870 train visual = 0.1050  # 单点/窗口波动
step880 train visual = 0.0720
step890 train visual = 0.0371
step900 train visual = 0.0351
step900 train pose   = 0.00121
step900 eval visual  = 0.2102
step900 eval pose    = 0.00658
last10 train visual mean   = 0.0535
last10 train visual median = 0.0391
last20 train visual mean   = 0.0505
last20 train visual median = 0.0391
last30 train visual mean   = 0.0513
last30 train visual median = 0.0403
last50 train visual mean   = 0.0502
last50 train visual median = 0.0411
speed window = 11.07 s/step
```

判断：step900 train visual 仍低，last10/20/30/50 median 也稳定低位；
但 eval900 出现一次显著高点 0.2102，需要视为风险信号而不是忽略。由于前面
eval600/750/850 均较低，且 train recent 没有同步恶化，当前更像随机 eval
window spike。最终判断必须等待 eval950/eval1000、fixed-seed eval 和
shared-tail gate summary。

2026-08-18 04:30 shared-tail step950/eval950：

```text
step910 train visual = 0.0385
step920 train visual = 0.0295
step930 train visual = 0.0543
step940 train visual = 0.0279
step950 train visual = 0.0418
step950 train pose   = 0.00060
step950 eval visual  = 0.0396
step950 eval pose    = 0.01794
last10 train visual mean   = 0.0471
last10 train visual median = 0.0378
last20 train visual mean   = 0.0461
last20 train visual median = 0.0378
last30 train visual mean   = 0.0502
last30 train visual median = 0.0403
last50 train visual mean   = 0.0492
last50 train visual median = 0.0411
speed window = 11.08 s/step
```

判断：eval950 visual = 0.0396，恢复到 eval600/750 的低区间，基本确认
eval900 的 0.2102 是随机 eval window spike，而不是结构性退化。train recent
在 900–950 也持续低位。最终仍需 step1000 checkpoint、fixed-seed eval
和 shared-tail gate summary。

2026-08-18 04:42 shared-tail final gate 完成：

```text
run = formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124
checkpoint = log/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124/checkpoints/step_001000.pt
checkpoint size = 8.56 GB

step1000 train visual = 0.0883
step1000 train pose   = 0.00427
step1000 eval visual  = 0.0407
step1000 eval pose    = 0.00317

last10 train visual mean   = 0.0810
last10 train visual median = 0.0480
last20 train visual mean   = 0.0673
last20 train visual median = 0.0415
last10 train pose mean     = 0.00135

fixed-seed eval:
  eval/loss_visual_velocity_mse = 0.0852
  eval/loss_pose_mse            = 0.00330
  eval/latent_x0_mse            = 0.0389
  eval/latent_x0_rmse           = 0.1798
  eval/latent_x0_l1             = 0.1297
  eval/baseline_noisy_latent_mse = 1.0360
  eval/latent_mse_improvement_vs_noisy = 0.9971
  eval/latent_x0_cosine         = 0.9682
  eval/pose_translation_mae     = 0.0273
  eval/pose_rotation_angle_deg  = 4.49
  eval/pose_fov_mae             = 0.0848

shared_tail_gate_summary:
  passed = true
  train_converged_by_robust_recent_stats = true
  max_recent_train_visual_robust = 0.18
  max_last_eval_visual_disaster_check = 0.6
```

artifact 路径：

```text
fixed eval:
result/v1_stage2_iw_aligned_re10k_pose_video/fixed_rnull_shared_tail_combo_schema_step1000_fixed_eval/eval_metrics.json

gate report:
result/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gates/formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124/shared_tail_gate_report.json

gate summary:
result/v1_stage2_iw_aligned_re10k_pose_video/formal_v1_gates/formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124/shared_tail_gate_summary.json
```

对比 fixed-rnull + legacy move/view baseline：

```text
baseline last10 train visual mean = 0.1398
candidate last10 train visual mean = 0.0810
baseline last_eval_visual = 0.0912
candidate last_eval_visual = 0.0407
paired mean visual diff candidate - baseline = -0.1160
```

最终判断：latest shared-tail fixedpatch 结构完成 1000-step formal gate，且
`passed=true`。虽然 step900 eval 和 step960 train 出现过 spike，但 eval950
和 eval1000 均恢复到低 visual 区间；robust recent stats、fixed-seed eval
和与 baseline 的 paired comparison 均支持“最新结构可稳定训练并优于 legacy
move/view 条件分支”的结论。

## 2026-08-21：Stage2 layer probe 与轨迹可视化检查

背景：当前 Stage2 final cotrain 的内置 `FramePoseHead` 从 current `Z_obs`
prefix hidden 读取 pose，但该 head 会对空间 token 做 mean pooling。为了确认
3D 信息是否已经存在于 backbone/register 表征中，新增 frozen-backbone layer
probe sweep：冻结完整 Stage2 模型，只在不同 Wan block hidden state 上训练
VGGT-style camera-query pose probe。

probe sweep 设置：

```text
checkpoint = log/v1_stage2_final_cotrain/stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/checkpoints/step_002000.pt
data       = RE10K, history_iw=1
layers     = 4,8,12,16,20,24,29
steps      = 500
probe      = DenseCameraQueryPoseProbe
readout    = per-frame camera query cross-attention over spatial tokens
```

最终 eval step500 排名：

```text
layer 16  pose_mse 0.003007  trans 0.04338  rot 7.062°  fov 0.05517
layer 12  pose_mse 0.003428  trans 0.04498  rot 6.983°  fov 0.06416
layer 29  pose_mse 0.004056  trans 0.05877  rot 3.921°  fov 0.07875
layer  8  pose_mse 0.004255  trans 0.05197  rot 8.201°  fov 0.07330
layer 20  pose_mse 0.004929  trans 0.08013  rot 7.493°  fov 0.06098
layer  4  pose_mse 0.005048  trans 0.04533  rot 10.609° fov 0.08123
layer 24  pose_mse 0.005203  trans 0.05470  rot 6.698°  fov 0.08847
```

结论：layer16 是当前 frozen probe 中最优读出层。probe 的 `pose_mse`
显著低于此前直接挂在模型里的 attached pose head 的 eval 量级，说明当前
backbone/register 中已有可读 3D 信息；主要问题更可能是正式 3D head 的读层
和读出结构，而不是表征中完全没有几何。

轨迹可视化逻辑检查：

- Stage2 pose target 格式确认为 `[tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]`。
- quaternion error 使用 pred/GT 的同一 `wxyz` 排布做 normalized absolute dot，
  计算 rotation angle，逻辑正确。
- 轨迹图画的是当前 Stage2 监督目标中的 relative transform translation
  三维分量，用于检查 pred/GT 对齐；它不是额外换算后的 global/absolute camera
  center。
- 修正 `scripts/visualize_v1_stage2_pose.py` 中原先错误的格式说明，并新增
  `--probe-checkpoint/--probe-layer`，支持同一批样本同时可视化 attached head
  与 frozen probe head。
- 新增 shared-axis comparison 图，避免不同 head 的 3D 轨迹图各自 autoscale
  造成肉眼误判。

本次可视化：

```text
output = result/v1_stage2_pose_visualization/step2000_attached_vs_probe_l16_20260821_trajectory_check
heads  = attached_head, probe_layer16
samples = 4
```

4 个样本 aggregate：

```text
attached_head:
  translation_l2 = 0.08630
  rotation_deg   = 7.28295
  fov_abs        = 0.03575
  pose_mse       = 0.002637

probe_layer16:
  translation_l2 = 0.09823
  rotation_deg   = 6.88045
  fov_abs        = 0.03959
  pose_mse       = 0.002087
```

小样本可视化结论：probe_layer16 在整体 pose MSE 上更优，但 translation/FOV
分量不保证每个样本都优于 attached head。sample000 的 GT 四帧几乎重合，
说明该 sampled window 的真实运动很小；pred 主要表现为整体偏置/漂移。因此
后续正式 head 改造时，建议优先采用 layer16 + camera-query spatial readout，
同时增加更大样本的 trajectory 分布统计，避免只看少量低运动窗口。

## 2026-08-21：启动两路后续训练

根据 layer probe 结果和 Stage3 数据准备现状，新增两条正式训练入口：

1. `scripts/train_v1_stage2_final_probe_layer16_cotrain.py`
   - 从当前 final Stage2 checkpoint 继续初始化：
     `log/v1_stage2_final_cotrain/stage2_final_branchmask_policyreg_venv_iw14816_20260819_015948/checkpoints/step_002000.pt`
   - 保留原 video generation / Register / action interface。
   - 将正式 3D head 改为 `DenseCameraQueryPoseHead`。
   - pose readout 从 Wan block layer16 hidden 读取 current `Z_obs` token。
   - 旧 mean-pooling `FramePoseHead` 权重不加载，新 pose head 随机初始化；
     backbone/register/action/video 权重从 step2000 加载。

2. `scripts/train_v1_stage3_final_vln_cotrain.py`
   - 从同一个 final Stage2 step2000 checkpoint 继续。
   - 使用已经准备好的 VLN T4 micro-latents：
     `/sharedata/NAV/derived/v1/vln/t4_micro_latents_500g/rxr_budget500_stream_20260821_0000/manifests/encoded_episodes.jsonl`
   - 当前可用数据约：
     `episodes ≈ 4901`，`history_iw=1` 可构造 `windows ≈ 14045`。
   - policy forward 中不输入 `A_cur`；policy branch 读取
     `Register + Z_obs + text/empty + A_noise`。
   - 更正：Stage3 正式 VLN cotrain 必须接入 instruction embedding，不能使用
     empty UMT5 作为正式训练条件。之前 empty-text run 只作为链路 smoke，
     不作为正式 Stage3 结果。
   - loss：

```text
L_stage3 =
  lambda_ce * CE(combo_logits, action_combo)
  + lambda_video_replay * L_visual
  + lambda_pose_replay  * L_pose
```

smoke test：

```text
stage2_probe_layer16 smoke:
  run = log/v1_stage2_final_probe_layer16_cotrain/smoke_stage2_probe_l16_1step_20260821
  ebs = 16
  seconds/step ≈ 100.05
  cuda_max_memory ≈ 28.10 GB
  train/loss_visual ≈ 0.0628

stage3_final_vln smoke:
  run = log/v1_stage3_final_vln_cotrain/smoke_stage3_final_vln_1step_20260821_b
  ebs = 16
  seconds/step ≈ 172.56
  cuda_max_memory ≈ 28.10 GB
  train/loss_policy = train/loss_ce_aux ≈ 5.47
  train/loss_video_replay ≈ 0.0794
```

注意：Stage3 每个 grad accumulation micro-batch 包含一次 policy forward/backward
和一次 Stage2 replay forward/backward，因此单步时间约为 Stage2 的 1.7 倍。

2026-08-21 23:30 修正：

- 新增 `scripts/cache_v1_stage3_vln_text_embeddings.py`。
- 从 VLN T4 latent sidecar 定位 raw-policy action json，读取 `instruction`，
  使用 Wan2.1 UMT5 encoder 缓存 per-sample instruction embedding。
- 默认输出：

```text
/sharedata/NAV/derived/v1/vln/text_embeddings/t4_micro/{dataset}/{sample_id}.pt
```

- 缓存脚本同时写出已完成 text cache 的 manifest snapshot：

```text
/sharedata/NAV/derived/v1/vln/text_embeddings/t4_micro/_cached_manifest_latest.jsonl
```

- `scripts/train_v1_stage3_final_vln_cotrain.py` 已改为从
  `--vln-text-cache-root` 读取每个 VLN sample 的 instruction embedding。
- 正式 Stage3 重启时必须使用 `--require-vln-text-cache`，避免静默 fallback
  到 empty text。
- 当前处理流程：

```text
1. 停止旧 Stage3 empty-text run。
2. GPU0 后台缓存 VLN instruction UMT5 embeddings。
3. 缓存完成后，用 _cached_manifest_latest.jsonl + --require-vln-text-cache
   重启 Stage3 final VLN cotrain。
4. GPU1 的 Stage2 layer16-probe cotrain 保持运行。
```

## 2026-08-25：Stage2 step3400 / Stage3 step400 最新权重评测

- Stage2 已按要求停止，最新 checkpoint 为 `step3400`；Stage3 训练保持运行，评测使用当时最新完整 checkpoint `step400`。
- evaluator 已统一兼容 Stage2 `data_config` 与 Stage3 `stage2_replay_data_config`，并在模型加载后重新固定 seed，保证两版使用相同窗口、timestep 与 diffusion noise。
- Mixed 64-batch 配对评测：Stage2/Stage3 latent x0 MSE 分别为 `0.02851/0.03542`，Stage3 高 `24.2%`。
- RE10K 64-batch pose：Stage2/Stage3 translation MAE 为 `0.04699/0.03814`，rotation error 为 `7.740°/7.368°`；Stage3 pose 未退化，但生成 latent MSE 更高。
- 4条完整30步生成：Stage2/Stage3 平均 RGB PSNR 为 `21.320/20.091 dB`；Stage3 下降 `1.229 dB`，逐帧可见更多纹理模糊和涂抹。所有视频均为有效 H.264、896×448、13帧。
- Stage3 R2R train open-loop：自然分布正确 instruction 下 accuracy 为 `58.75%`（random A_noise）或 `60.16%`（zero A_noise），多数类基线为 `60.08%`；TURN_LEFT/RIGHT recall 均为0。打乱 instruction 的 paired top-1 disagreement 仅 `0.86–1.48%`，说明 step400 尚未有效利用 instruction。
- 完整报告与配对帧位于 `result/v1_stage2_stage3_comparison/paired_step3400_vs_step400_seed20260827/`。
- 解释限制：Stage3 从 Stage2 step2600 分叉，而比较的当前 Stage2 已到 step3400；此结果反映当前两条分支的实际差异，不能将全部差值直接归因为 Stage3 遗忘。

## 2026-08-26：Stage3 单动作四分类均衡诊断

本实验不提前修改 VLN observation 的时间粒度，只验证当前完整共享模型是否能在
更直接的离散监督下学出有效动作预测；因此它是 `diagnostic`，不是对最终
Stage3 时间接口的重新定版。

```text
source checkpoint:
  Stage2 step3400

完整输入与共享模型：
  K=1..7 T4 history + A_hist -> Register
  Register + current T4 Z_obs + instruction + one fixed action query
  single shared WanBlock backbone

policy readout:
  final action hidden [B,1,1536]
  -> FP32 LayerNorm
  -> Linear(1536,512) + GELU + Linear(512,4)
  -> logits [B,1,4]

target:
  one next action in {STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT}

sampling:
  physical BS=1, grad_accum=16, EBS=16
  every four consecutive policy samples contain each class exactly once
  minority examples are sampled with replacement; latent files are not copied

loss:
  L = CE_4(next_action)
    + 0.25 * L_visual_replay
    + 0.05 * L_pose_replay

optimizer:
  shared backbone/Register/video/pose: bf16 forward, lr=2e-6
  fresh policy head + action query input modules: FP32, lr=1e-4
```

现有单动作锚点为 `88,602` 个，天然分布为：

| action | anchors | natural ratio | effective train ratio |
| --- | ---: | ---: | ---: |
| STOP | 915 | 1.03% | 25% |
| MOVE_FORWARD | 63,527 | 71.70% | 25% |
| TURN_LEFT | 11,931 | 13.47% | 25% |
| TURN_RIGHT | 12,229 | 13.80% | 25% |

代码入口：

```text
src/nav/v1/stage3_single_action.py
scripts/train_v1_stage3_r2r_single_action.py
```

正式诊断 run：

```text
Stage3 GPU1:
  log/v1_stage3_r2r_single_action/
    stage3_r2r_single_action_balanced_from_stage2step3400_2k_20260826/

Stage2 GPU0 continuation:
  log/v1_stage2_final_cotrain/
    stage2_from_step3400_continue_lr2e6_1k_20260826/
```

验收时不能只看 balanced accuracy；至少同时报告四类 recall、macro recall、
自然分布 accuracy/majority baseline 和 instruction shuffle/empty ablation。

启动核对：正式 Stage3 run 的第一个完整 optimizer step 已完成，包含16次真实
R2R policy forward/backward 与16次 Stage2 replay forward/backward：

```text
step1 seconds/step = 140.49
cuda_max_memory    = 28.23 GiB
target counts      = STOP/MOVE/LEFT/RIGHT 各4
policy CE          = 1.3964（随机四分类参考 ln(4)=1.3863）
action accuracy    = 12.5%
instruction cache  = 16 hits / 0 fallback
```

首步只证明完整链路、类别配额和监督形状正确，不作为收敛效果结论。
