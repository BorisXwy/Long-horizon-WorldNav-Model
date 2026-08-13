# V1 DreamZero Wan2.2-5B Backbone 使用方式与延迟记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-ARC-003` |
| 类型 | 架构解析 / Latency Probe |
| 状态 | Verified / Synthetic Core Probe |
| 更新时间 | 2026-08-08 |
| 职责 | 记录 DreamZero 官方 Wan2.2-TI2V-5B 路径如何使用 timestep、action/state register、KV cache，以及本机核心 DiT 延迟 |

## 当前结论

DreamZero 5B 路径不是 one-step generator。它默认使用 `16` 个 scheduler
timestep，但只在其中 `8` 个 timestep 真正调用 DiT，其他 timestep 复用上一次
`flow_pred`，因此可以理解为 **16-step solver / 8-step DiT compute**。

本机 RTX 6000 Ada 上，使用 DreamZero 官方 5B 架构、Wan2.2-TI2V-5B latent
规格、synthetic 输入测得核心 DiT/action-register loop：

| 口径 | 权重 | DiT计算步数 | 总时间 | 单个实际DiT step | 峰值显存 |
| --- | --- | ---: | ---: | ---: | ---: |
| 官方 mask | random 结构权重 | 8 / 16 | 0.81s | 约0.10s量级 | 10.57GiB |
| 官方 mask | base Wan2.2 DiT 权重 | 8 / 16 | 0.99s | 约0.11–0.13s | 10.57GiB |
| 全量上界 | random 结构权重 | 16 / 16 | 1.17s | 约0.073s | 10.57GiB |
| 全量上界 | base Wan2.2 DiT 权重 | 16 / 16 | 1.34s | 约0.083s | 10.57GiB |

统计口径：只测 DreamZero 5B `CausalWanModel` 的 synthetic latent + text context +
CLIP context + noisy action/state register 前向循环，不包含 T5、CLIP image encoder、
VAE encode/decode、websocket server、真实 checkpoint metadata 和环境交互。

## 官方 5B 路径与输入结构

DreamZero 的官方 5B 配置见：

- 代码：`/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/dreamzero`
- 5B说明：`dreamzero/docs/WAN22_BACKBONE.md`
- 训练入口：`dreamzero/scripts/train/droid_training_wan22.sh`
- 核心 action head：`dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- Causal DiT：`dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

官方 Wan2.2-5B 配置要点：

| 项 | DreamZero 5B 设置 |
| --- | --- |
| Base Backbone | `Wan2.2-TI2V-5B` |
| DiT dim / layers / heads | 3072 / 30 / 24 |
| VAE latent channels | 48 |
| 视频分辨率 | 160×320 |
| latent spatial | 10×20 |
| patch 后 tokens per latent frame | 50 |
| `num_frame_per_block` | 2 |
| `num_action_per_block` | 24 |
| `action_horizon` | 24 |
| action/state register | 每个 block 拼接 `24` 个 action tokens + `1` 个 state token |

它将 DiT 输入组织为：

```text
[video tokens | action/state register]
```

其中 action register 不是外部 head 单独预测，而是进入同一个 Wan DiT transformer
序列。DiT 同时输出：

- `video_noise_pred`：视频 latent 的 flow/noise prediction；
- `action_noise_pred`：从 action-register slice 经 `action_decoder` 得到的 action
  noise prediction。

## 完整模型结构示意图

下面示意图描述的是 DreamZero 官方 Wan2.2-5B 路径的一次 closed-loop chunk
推理，不包含外部 robot environment 的真实执行耗时。图中 `Video Latent` 与
`Action/State Register` 在同一个 DiT 序列中联合去噪；历史 chunk 主要通过
`KV Cache` 进入后续 block，而不是像 NAV 计划中那样压缩进长期 Register。

```text
                         DreamZero Wan2.2-5B Causal DiT policy backbone

                    ┌──────────────────┐    ┌──────────────────────┐
                    │ Language          │    │ Current / first      │
                    │ instruction       │    │ visual observation   │
                    └────────┬─────────┘    └──────────┬───────────┘
                             │                         │
                             ▼                         ▼
                    ┌──────────────────┐    ┌──────────────────────┐
                    │ Text Encoder      │    │ CLIP Image Encoder   │
                    │ text context      │    │ image context        │
                    └────────┬─────────┘    └──────────┬───────────┘
                             │                         │
                             │       ┌─────────────────┴─────────────────┐
                             │       │         DiT conditions             │
                             │       │ text / image / timestep / position │
                             │       └─────────────────┬─────────────────┘
                             │                         │
                             │                         ▼
                             │       ┌───────────────────────────────────┐
                             │       │  16-step Flow scheduler            │
                             │       │  default: 8 / 16 DiT compute mask  │
                             │       └─────────────────┬─────────────────┘
                             │                         │
                             ▼                         ▼

┌───────────────────────────────────────┐      ┌────────────────────────────────────────────────┐
│ Main noisy input sequence              │      │ CausalWanModel                                  │
│                                        │      │ Wan2.2-TI2V-5B DiT + DreamZero extensions        │
│  Observation block O_k                 │      │                                                │
│      │                                 │      │  ┌──────────────────────────────────────────┐  │
│      ▼                                 │      │  │ 30-layer Wan DiT Transformer              │  │
│  Wan VAE Encoder                       │      │  │ dim=3072, heads=24                        │  │
│      │                                 │      │  │ causal block attention                    │  │
│      ▼                                 │      │  │ cross-attn to text/image conditions       │  │
│  noisy video latent x_t^video          │      │  └──────────────────────────────────────────┘  │
│      │                                 │      │                                                │
│      ▼                                 │      │  history input from top:                      │
│  Patchify video latent                 │      │      KV Cache of previous causal chunks        │
│  50 tokens / latent frame              │      │                                                │
│                                        │      └──────────────────────┬─────────────────────────┘
│  Noisy action chunk a_t                │                             │
│      │ 24 actions                      │                             │
│      ▼                                 │                             ▼
│  action_encoder                        │      ┌────────────────────────────────────────────────┐
│      │                                 │      │ Split output token slices                       │
│      ▼                                 │      └───────────────┬──────────────────────┬─────────┘
│  24 action tokens                      │                      │                      │
│                                        │                      │                      │
│  Robot state                           │                      ▼                      ▼
│      │                                 │      ┌────────────────────────┐   ┌──────────────────────┐
│      ▼                                 │      │ video prediction head   │   │ action_decoder       │
│  state_encoder                         │      │ video_noise_pred        │   │ action_noise_pred    │
│      │                                 │      │ / flow_pred             │   │                      │
│      ▼                                 │      └────────────┬───────────┘   └──────────┬───────────┘
│  1 state token                         │                   │                          │
│                                        │                   ▼                          ▼
│  concat as DiT input:                  │      ┌────────────────────────┐   ┌──────────────────────┐
│  [video tokens | action tokens | state]├─────►│ Video scheduler update │   │ Action scheduler      │
└───────────────────────────────────────┘      │ x_t^video              │   │ update a_t            │
                                               └────────────────────────┘   └──────────┬───────────┘
                                                                                       │
                                                                                       ▼
                                                                            ┌──────────────────────┐
                                                                            │ Predicted action     │
                                                                            │ chunk: 24 actions    │
                                                                            └──────────┬───────────┘
                                                                                       │
                                                                                       ▼
                                                                            ┌──────────────────────┐
                                                                            │ Execute in env        │
                                                                            │ observe O_{k+1}       │
                                                                            └──────────────────────┘

  KV Cache path:
      previous chunks ───────────────────────────────► CausalWanModel ───► update KV Cache for k+1
```

如果用更紧凑的序列视角看，DreamZero 的 DiT 输入是：

```text
for each causal block:
  video_tokens = patchify(noisy_video_latent)
  action_tokens = action_encoder(noisy_action_chunk)
  state_token = state_encoder(robot_state)

  dit_tokens = concat(video_tokens, action_tokens, state_token)
  dit_output = CausalWanModel(
      dit_tokens,
      timestep,
      text_context,
      clip_image_context,
      kv_cache_from_previous_blocks,
      current_start_frame,
  )

  video_noise_pred = video_head(dit_output.video_slice)
  action_noise_pred = action_decoder(dit_output.action_slice)
```

## Timestep 使用方式

在 `WANPolicyHead.__init__` 中，官方代码固定：

```text
num_inference_steps = 16
sigma_shift = 5.0
```

推理时使用两个 `FlowUniPCMultistepScheduler`：

- video scheduler：更新 noisy video latent；
- action scheduler：更新 noisy action chunk。

二者各自设置 `16` 个 timestep。每个 scheduler timestep 都会更新 sample，但并非
每个 timestep 都调用 DiT。默认 `NUM_DIT_STEPS=8` 时，mask 为：

```text
[T, T, T, F, F, F, T, F, F, F, T, F, F, T, T, T]
```

其中 `T` 表示调用 DiT，`F` 表示复用上一次 prediction。若设置
`NUM_DIT_STEPS=16` 或其他非 5/6/7/8 值，则 16 个 timestep 都调用 DiT。

这和我们之前测 InfiniteWorld / NAV 的区别非常关键：

- InfiniteWorld 官方是 `30` 个 sampling steps，基本每 step 都跑 DiT；
- DreamZero 5B 是 `16` 个 solver steps，但默认只有 `8` 个 DiT compute steps；
- 因此 DreamZero 的“快”主要来自短 latent block、small 5B backbone、KV cache
  和 DiT step skipping，而不是 one-step denoising。

## KV cache 与闭环推理

DreamZero 5B 的闭环策略是：

```text
observe one block -> predict one action chunk -> execute -> observe next block -> reuse KV cache
```

`current_start_frame` 记录当前 block 在长序列中的位置。第一次请求会 warm up
第一帧和 KV cache；后续请求只处理新 block 的 video tokens + 新 action/state
register，历史 block 通过 KV cache 保留，不重复完整前向。

这与 NAV 当前 Register 方案的差别：

- DreamZero 的 long history 是 KV cache / blockwise causal attention；
- NAV Stage One 的目标是将 long history 压进 Register，而不是无限增长 KV；
- 但 DreamZero 的 action/state register 拼接进 DiT 序列这一点，对 NAV Stage
  Three 的 navigation action head 很有参考价值。

## 本地执行记录

脚本：

```bash
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/benchmark_dreamzero_wan22_5b_latency.py
```

环境：

```bash
source /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_wan2_2/bin/activate
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/dreamzero
```

官方 mask + base Wan2.2 权重命令：

```bash
TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
python /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/benchmark_dreamzero_wan22_5b_latency.py \
  --device cuda:0 \
  --warmup 1 \
  --repeat 1 \
  --steps 16 \
  --dit-compute-steps 8 \
  --load-weights \
  --out /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/dreamzero_5b/wan22_5b_latency_loaded_mask8.json
```

结果文件：

| 结果 | 路径 |
| --- | --- |
| random / 8 DiT mask | `NAV/result/dreamzero_5b/wan22_5b_latency_random_mask8.json` |
| loaded Wan2.2 / 8 DiT mask | `NAV/result/dreamzero_5b/wan22_5b_latency_loaded_mask8.json` |
| random / 16 DiT上界 | `NAV/result/dreamzero_5b/wan22_5b_latency_random.json` |
| loaded Wan2.2 / 16 DiT上界 | `NAV/result/dreamzero_5b/wan22_5b_latency_loaded.json` |

## 权重兼容性

使用 `/sharedata/Wan2.2-TI2V-5B` 的 DiT safetensors 加载到 DreamZero
`CausalWanModel`：

- loaded keys：825；
- unexpected keys：0；
- missing keys：172。

missing key 主要包括：

- DreamZero 新增 `state_encoder`；
- DreamZero 新增 `action_encoder` / `action_decoder`；
- 部分 action/register 扩展后的 image cross-attn 参数。

这说明 DreamZero 5B 路径确实是从 Wan2.2 backbone 继承主体权重，再增加
action/state register 相关结构；base Wan2.2 权重本身不包含完整 policy 能力。

## 限制与未决问题

1. 当前没有下载官方 DreamZero-DROID 14B 或 DreamZero-AgiBot checkpoint，也没有
   官方训练好的 DreamZero-5B policy checkpoint；因此本记录不是 RoboArena 或
   DROID sim 的 policy score 测评。
2. 当前 latency 是核心 DiT/action-register latency，不包含 T5/CLIP/VAE/server。
3. 本机是 RTX 6000 Ada，非 DreamZero README 中主要优化的 GB200/H100 +
   TensorRT/TE 路径；官方 README 的 0.6s/3s 不能与本结果直接横向比较。
4. 若后续要做真实 closed-loop policy latency，需要准备完整 DreamZero checkpoint
   目录（含 `experiment_cfg/conf.yaml`）并运行 `eval_utils/serve_dreamzero_wan22.py`。
