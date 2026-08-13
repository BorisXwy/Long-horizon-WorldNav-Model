# V1 Stage One T4-IW 对齐训练规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-007` |
| 类型 | 训练规范（Training Specification） |
| 状态 | T4/IW Data Rule Active；Old Wan Training Entry Superseded |
| 更新时间 | 2026-08-14 |
| 职责 | 固定 V1 Stage One 中 `T_latent=4` 的数据切片、history 构造和 InfiniteWorld 对齐口径 |

## 当前结论

V1 Stage One 的新版默认 generation horizon 采用 `T_latent=4`，即每次只对一个短
future latent segment 做 RFlow / diffusion loss，而不是恢复 InfiniteWorld 的
81-frame dense target。这样保留“noisy target token 小、训练/推理更快”的核心收益。

**2026-08-14 更新**：本文中的 T4 micro chunk、IW-1/4/8/16 history 换算、
latent manifest 与 window 构造规则仍是当前有效规则；但 2026-08-13 版本的
`A_query` / single hidden-width action token 实现已被 DEC-034 覆盖。新的正式
action interface 使用 `A_noise -> action flow decoder -> A_out`、
`H_action=H_nav=10` 和 dual-stream / MoT-style backbone。旧 Wan 训练入口已从
当前代码删除；当前可执行结构验证见 `NAV-EVL-005`。

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

## 数据整备

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

## 训练样本构造

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
  `A_noise` tokens 进入 action stream / shared-coupled DiT backbone；
  `A_out` 从 action stream hidden 经 Giga/DreamZero-style action flow decoder
  读出。
  Stage One 中 `lambda_action=0`，不计算 policy/action loss。
```

因此 Stage One 只训练 video generation memory，不把 video pseudo action 训练成
policy。真正的 action output supervision 在 Stage Three 用 VLN 数据训练。
旧版 `Register/local -> 旁路小 action head` 是错误 scaffold，已废弃，不能作为
正式 Stage One 或 policy latency 结论。

## Text-conditioned fullmix 5000-step 正式训练（2026-08-12）

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

## 训练入口状态

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

## 当前限制

- 当前正式 Stage One 训练脚本采用 GigaWorld-Policy-style per-token timestep：
  clean prefix / Register / local visual tokens 为 `t=0`，future visual 使用
  RFlow `visual_t`，`A_noise` action tokens 使用独立 `action_t`。现有代码仍需
  从旧 `A_query` smoke 实现迁移到 DEC-034 的 `A_noise` + dual-stream action
  flow decoder。
- 正式 long-history 已覆盖 IW-16：107 个 T4 micro history updates，target 是
  第 108 个 micro chunk，用于对齐 InfiniteWorld 的 16-chunk teacher history。
- Stage One action output branch 只保留格式，不训练 policy/action loss；
  Stage Three 才训练 action output。
- 当前 T4 route 是 Stage One generation branch；Stage Two 3D probe 和 Stage
  Three policy-only 裁剪尚未在该脚本中实现。

## Giga timestep 历史 smoke（2026-08-13）

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
  # historical smoke only; formal DEC-034 uses A_noise + dual-stream action decoder

loss =
  L_visual_flow only
  lambda_action = 0

smoke result =
  loss = 1.1692
  visual_timestep_mean = 917.26
  action_timestep_mean = 998.0
  peak_reserved_gib = 30.95 GiB
```

## 旧 fullmix 训练计划（2026-08-13，已被 DEC-034 覆盖）

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

## 本轮结束后的测评与续训顺序

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
