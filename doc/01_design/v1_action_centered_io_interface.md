# V1 NAV action-centered 主输入与输出接口设计

| 字段 | 内容 |
| --- | --- |
| ID | NAV-DES-004 |
| 类型 | 设计规范 |
| 状态 | Accepted Design / Not Implemented |
| 更新时间 | 2026-08-14 |
| 职责 | 定义 NAV 新版 GigaWorld-like + Register backbone 的 DiT 主输入、condition、输出解码、控制输出，以及它们与视频数据、导航数据、时间/位移尺度的对应关系 |

## 核心结论

新版 NAV 不应再把完整 81-frame future video latent 当作策略主输入。推荐接口是：

```text
主输入 main denoised variables:
  1) noised future visual latent        只在 Stage One/Two 训练和视频评测时存在
  2) noised action/nav tokens           Stage One/Two/Three 都存在，是 policy 要生成的主变量

video/main context tokens:
  1) current/local observation latent   来自当前 RGB 或当前短窗口 RGB
  2) history/Register video tokens      来自 extractor/update，不由 future GT 初始化

condition:
  1) Instruction/Text condition
  2) diffusion/flow timestep condition
  3) modality/time/position/dataset/action-scale embedding
  4) optional state/previous-action/embodiment condition

输出:
  1) future visual velocity/noise       解码为视频 consequence
  2) action/nav velocity/logits         直接转成控制
  3) selected hidden/register features  Stage Two 接 3D probe，Stage Three 接导航头
```

这保持 Action-centered 因果原则：`history/current obs/instruction -> action -> future visual/3D consequence`。future visual 和 future 3D 是 consequence supervision，不是 policy condition。

Action token 的语义必须特别区分：

```text
policy side:
  不存在已知 current/target action input。
  action/nav tokens 统一为 A_noise，是要被模型生成/去噪的变量。

generation branch side:
  可以读取 DiT backbone 内的 action stream hidden state。
  目的是真正建模 “在这个 action 下，future visual consequence 应该如何变化”。

history side:
  Register 由历史/current visual observation 和已经执行过的 A_hist 更新；
  A_hist 必须是独立 action tokens / cross-attention context，
  不能用 action bias / latent bias 方式注入。
```

### 三类 action 的硬性区分

V1 Stage One 的最终结构必须同时保留三类 action，它们语义不同，不能合并成同一个
`move/view condition`：

```text
1. history action / previous executed action
   记号: A_hist_i
   来源: 已经发生的历史 chunk C_i 对应的 motion/action
   用途: 与历史 video chunk 一起参与 Register update，帮助解释历史视觉变化
   形式: action tokens / cross-attention context / gated token adapter
   禁止: action bias / latent bias / 直接加到 visual latent 或 Register tokens
   因果性: 可作为输入，因为它是过去已经执行/观测到的事件。

2. current action for video generation
   记号: A_cur
   来源: 目标 future chunk 对应的 action；视频生成训练时可用 GT/pseudo action
   用途: 作为 generation branch 的 DiT condition，控制 future visual consequence
   形式: DiT condition / action condition tokens
   因果性: 只服务“给定 action 生成 consequence”；不是 policy side 的输入。

3. noised action latent / action output
   记号: A^σ -> A_out
   来源: Stage One/Two 训练时由 target action chunk 加噪/随机 mask 得到；
         Stage Three 推理时由 action prior/noise 初始化得到。
   用途: action/policy branch 的主变量，最后输出 action chunk / nav action logits
   形式: action stream tokens + Giga/DreamZero-style action flow decoder
   因果性: 它是要生成的变量；不能直接读取 clean A_cur，也不能读取 future visual。
```

视频生成训练时三者同时存在：

```text
history update:
  R_{-1} = R_null  # fixed, non-episode-specific
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  # A_hist 以 token/cross-attention 方式交互；禁止 bias 注入

video generation branch:
  input:  R_t, local/current obs, Z_future^σ
  cond:   text/instruction, timestep, A_cur
  output: visual velocity/noise for future chunk

action branch:
  input:  R_t, local/current obs, A_noise
  cond:   text/instruction, action timestep/type embedding
  output: action velocity/noise -> denoise/decode -> A_out
  loss:   Stage One 默认不启用 policy/action loss；只保留 token slot、mask 和 head
          格式。Stage Three 才用 VLN action label 训练 A_out。
```

Stage Three 导航推理时没有 `A_cur` 输入；只保留：

```text
R_t + current obs + instruction + A_noise -> action velocity/noise -> A_out
```

因此 Stage One 代码检查项：

```text
必须使用统一 RegisterCell，而不是单独 Extractor / Updater；
RegisterCell 必须读取 fixed R_null/R_prev、history/current visual tokens 与 A_hist tokens；
禁止使用 action bias / latent bias 方案；
必须有 current action 控制 video generation；
必须有 noised action tokens 和 action flow decoder；
action branch 不允许从 future visual token 泄露信息。
```

其中“必须有”指接口和 forward path 必须存在；Stage One 视频数据上不要求
`A_out` 具备 policy 能力，也不把 pseudo action 当作主要监督。若没有可靠 action
或 text 标注，history/current action 可使用 empty/unknown embedding；Stage One
仍主要优化 video diffusion / flow loss。

## 记号和默认尺寸

### Wan2.1 / InfiniteWorld 兼容侧

当前项目基础仍是 Wan2.1 / InfiniteWorld 权重，因此 latent 侧先保持 Wan VAE 规则：

```text
RGB video:             [B, 3, T_rgb, H_rgb, W_rgb]
Wan latent:            [B, 16, T_lat, H_lat, W_lat]
temporal factor:       4
spatial factor:        8
patch size in DiT:     [1, 2, 2]  # 以现有 InfiniteWorld/Wan 风格为准
```

若继续使用我们此前 InfiniteWorld/VBench 的 896×448 尺度：

```text
H_rgb × W_rgb = 448 × 896
H_lat × W_lat = 56 × 112
1 个 latent time 的 visual tokens = 56/2 × 112/2 = 1568
```

原来的 81-frame chunk：

```text
T_rgb = 81
T_lat = (81 - 1) / 4 + 1 = 21
future visual tokens = 21 × 1568 = 32928
```

新版第一版建议改成 sparse future consequence：

```text
current/ref latent:    [B, 16, 1, 56, 112]  -> 1568 tokens
future noisy latent:   [B, 16, 1, 56, 112]  -> 1568 tokens
visual stream total:   3136 tokens
```

这比旧 81-frame dense future 少约 10.5 倍 visual token。若后续确认 Wan2.1 代码对分辨率完全兼容，可以再评估 448×448 或 384×320，以进一步降低 token 数；但第一版以 896×448 保持与现有权重、脚本和评测路径最大兼容。

### History/Register video tokens

History 仍然应该放在 video token / main stream 侧，而不是默认放在 text
condition 侧。原因是我们希望 history 与 current observation、noisy future
latent 在同一套 DiT self-attention / MoT 计算里交互；如果只作为
cross-attention condition，它更像静态提示，难以成为可持续更新的时空状态。

Register 是跨时间历史的紧凑 video token 状态，不再存储完整历史 latent：

```text
Register/history R_t:  [B, K, D_reg]
推荐初始 K:            128 或 256
D_reg:                 进入 DiT 前投影到 Wan2.1 hidden_dim
初始化策略:             R_{-1}=R_null fixed template；R_0=RegisterCell(R_null, concat([visual_tokens(C_0), A_hist0/no-op action tokens]))
更新策略:               R_t=RegisterCell(R_{t-1}, concat([visual_tokens(C_t), A_hist_t]))
```

关键语义：

- `R_null` 是固定初始 register template，不携带 episode-specific 信息，也不作为
  learnable scene prior。
- `R_0` 与后续 `R_t` 都由同一个 `RegisterCell` 在线更新得到；区别只在于
  第一次的 previous register 是 `R_null`。
- 后续 `R_t` 作为下一步的 history video tokens
  重新进入 visual/main stream。
- RegisterCell 权重同时承担“首步写入”和“后续更新”能力，但不依赖固定
  episode-specific 初始记忆。
- GT pose 不作为推理输入。Pose 只用于训练数据构造、pseudo action、3D supervision
  和数据审计；真实世界推理依赖 observation、instruction，以及已经执行过的
  previous action/motion token。`A_hist` 可以参与 Register 更新，但只能通过
  token/cross-attention 交互，不能作为 additive bias 写入 latent/Register。
  当前要生成的 action 不作为 policy 输入。

## 主输入 main input 怎样组成

这里的“主输入”指进入 DiT block 并被 diffusion/flow 预测的变量，不是所有条件。

### Stage One / Stage Two：视频生成 + action-centered cotrain

训练样本从视频数据构造：

```text
history/current observation:
  O_t 或短 local obs window

pseudo action:
  a_{t:t+H_a-1}
  由 pose 差分 / 插值 / 离散化获得

future visual consequence:
  F_{t+Δ}
  或 sparse frames [F_1, F_2, F_3, F_4]
```

主输入由两条流组成：

```text
visual stream:
  history/Register video tokens:
    R_t = RegisterCell(R_{t-1}, concat([visual_tokens(C_t), A_hist_t]))
    shape = [B, K, D_reg] -> projected to DiT hidden_dim

  clean current/ref latent:
    Z_ref = VAE(O_t)
    shape = [B, 16, 1, H_lat, W_lat]

  noised future latent:
    Z_future^σ = add_noise(VAE(future sparse frames))
    shape = [B, 16, T_future, H_lat, W_lat]

action stream:
  noised action tokens:
    A^σ = add_noise(action embedding)
    shape = [B, H_a, D_action]
```

Stage One/Two 训练中，`A^σ` 是 action stream 的主变量。Generation branch 可以
读 action stream 的 hidden state 来预测 `Z_future`，但 policy/action stream
本身不把 clean GT action 当 condition。

第一版推荐：

```text
T_future = 1
H_a = 10
```

这里 `H_a=10` 是统一 action horizon。视频数据侧把 pose/ego-motion/pseudo motion
采样或重采样到 10 个 action slots；VLN 侧也预测 10 个 low-level actions。这样
Stage One/Two/Three 的 action stream 形状保持一致，后续从视频生成切到导航策略时
不再更换 action token 数。

如果模仿 GigaWorld 的 sparse visual pack，可以用 5 个 RGB frame：

```text
[obs_t, future_1, future_2, future_3, future_4]
```

经过 Wan VAE temporal factor 4 后得到：

```text
T_lat = (5 - 1) / 4 + 1 = 2
clean ref latent T = 1
noisy future latent T = 1
```

因此 `T_future=1` 不表示只监督一个普通 RGB 帧，而是监督一个被 VAE 压缩后的 future consequence latent；它可以对应一组 sparse future frames。

当前命名约定：

```text
生成分支 generation branch:
  future visual branch / future consequence branch

Z_future T=1:
  one future latent time cell
  ≠ one RGB frame
  ≈ 当前 obs 后的 4 个 future RGB slots，在 sparse pack 中表示一组 future consequence
```

如果后续需要更长的生成分支：

```text
Z_future T=2:
  约对应 obs 后 8 个 future RGB slots

Z_future T=4:
  约对应 obs 后 16 个 future RGB slots
```

但第一版默认保留 `T_future=1`，先验证 small generation branch 是否足以提供动作导致未来变化的监督信号。

### Stage Three：VLN policy-only 推理

导航推理不生成视频时，future visual branch 删除：

```text
visual stream:
  history/Register video tokens:
    R_t
    shape = [B, K, D_reg] -> projected to DiT hidden_dim

  clean current/ref latent:
    Z_obs = VAE(current RGB or short obs window)
    shape = [B, 16, 1, H_lat, W_lat]

  noised future latent:
    empty, T_future = 0

action/nav stream:
  noised action tokens:
    A_noise
    shape = [B, H_nav, D_action]
```

推荐第一版导航 chunk：

```text
H_nav = 10 low-level steps
```

原因是 R2R-CE / RxR-CE 一类 Habitat action 的基础尺度通常是：

```text
MOVE_FORWARD ≈ 0.25 m
TURN_LEFT/RIGHT ≈ 30°
```

`H_nav=10` 不要求一次开环执行 10 步；它是 action generation horizon。闭环推理
默认仍执行第一个 action 后更新 observation/Register。设为 10 的动机是：动作支路
采用 Giga/DreamZero 常见的 noised action chunk decode 形式，较长 horizon 可以
提供更稳定的局部轨迹意图，同时保持 policy-only 推理只跑 action stream 和共享
backbone 的裁剪路径。

## condition 怎样组成

Condition 不应包含 future GT。推荐组成：

```text
Text/Instruction condition:
  video data:
    caption / scene text / action text / empty prompt
  VLN data:
    R2R/RxR/ScaleVLN instruction

Timestep condition:
  visual denoise timestep σ_v 或 t_v
  action denoise timestep σ_a 或 t_a

Modality/type condition:
  obs token / future token / action token / register token / text token

Temporal/position condition:
  visual latent time index
  action step index
  nav chunk index

State/embodiment condition:
  previous action  # 已执行历史 action，可作为 A_hist tokens；不是当前 target action
  optional camera id / robot id / dataset id
  optional action scale calibration
```

对 Wan2.1 / InfiniteWorld 兼容实现而言：

```text
Text condition:
  UMT5/T5 embedding
  shape = [B, L_text, D_text]

Register:
  不作为默认 condition。
  默认作为 visual/main stream 的 history video tokens。
```

如果短期工程上保留 B 方案作为 ablation，可以把 Register 投影后接到 text
condition 侧；但新版主方案应把 history/Register 放在 video/main stream。否则
Stage Two/Three 要取 `Register-after-DiT` 时，容易只拿到 cross-attention 前的
静态 embedding，而不是被 DiT 时空计算真正更新过的 history state。

## DiT 内部计算：双流主方案

Generation branch 和 policy/action branch 的 raw input 形式不同，不应强行拼成
同一种原始 tensor。V1 主方案采用：

```text
modality-specific stem
  + shared / coupled DiT blocks
  + modality-specific heads
```

也就是：

```text
visual/main stream:
  Register video tokens
  obs latent patch tokens
  future noisy latent patch tokens

action/policy stream:
  noised action tokens A_noise
  during training: noisy target action chunk
  during inference: Gaussian/noise prior action chunk

condition stream:
  instruction/text
  timestep
  modality/time/dataset/action-scale embedding
  optional previous executed action/state as A_hist tokens
```

推荐的计算图：

```text
                         text / timestep / type / scale condition
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Shared / Coupled DiT                        │
│                                                                      │
│  visual/main stream                         action/policy stream      │
│                                                                      │
│  [R_t][Z_obs][Z_future^σ]  ◄──── controlled interaction ────► [A_noise]│
│        │                                                        │     │
│        │                                                        │     │
│        ▼                                                        ▼     │
│  future visual head                                      action/nav head│
│  predict v/noise(Z_future)                               logits / action│
└──────────────────────────────────────────────────────────────────────┘
```

这里“共享”不是指两个 branch 的输入维度、token 数和输出头相同，而是指二者在
DiT blocks 内通过统一的 hidden space、条件调制和受控 attention/cross-attention
交互。

### Stem

Visual/main stream stem：

```text
Register:
  R_t [B,K,D_reg] -> linear/project -> [B,K,D_v]

Obs latent:
  Z_obs [B,16,1,H,W] -> patch_embed -> [B,N_obs,D_v]

Future latent:
  Z_future^σ [B,16,T_future,H,W] -> patch_embed -> [B,N_future,D_v]
```

Action/policy stream stem：

```text
Noised action variable:
  A_noise / A^σ -> action_embed -> [B,H_action=10,D_a]

Stage One/Two video batch:
  若没有要训练的 policy action loss，仍构造 A_noise slot；
  A_noise 可来自 no-op/empty action target 加噪或纯 noise prior；
  只用于固定 action stream 格式与 generation branch 交互。

Stage Three VLN batch:
  GT action chunk -> action embedding -> add_noise -> A_noise；
  action decoder 预测 action velocity/noise；
  denoise/decode 后得到 A_out。
```

`D_v` 和 `D_a` 可以相同，也可以类似 GigaWorld 使用不同宽度。正式结构要求保持
dual-stream / MoT-style 语义；工程实现如果为了兼容 Wan2.1 先让二者同 hidden_dim，
也必须保留独立 action expert/decoder、stream/type embedding 和 causal mask，不能
退化成无区分的旁路 action head。

### Causal attention / interaction mask

正式主方案把 tokens 分成 6 类，并按因果方向固定可见性：

| Query token | 可 attend | 不可 attend / 不可接收 |
| --- | --- | --- |
| `R_t` Register/history | `C_hist` summary、`C_local/Z_obs`、`A_hist` tokens、instruction/text、dataset/type/time embedding | `Z_future^σ`、future 3D GT、clean `A_cur`、clean target action、任何 additive action bias |
| `Z_obs` clean current/local visual | `R_t`、instruction/text、`A_hist` tokens、type/time embedding | `Z_future^σ`、future 3D GT、clean target action、任何 additive action bias |
| `A_noise` action stream | `R_t`、`Z_obs`、instruction/text、`A_hist` tokens、self action tokens、action timestep `t_a` | `Z_future^σ`、future 3D GT、clean `A_cur`、clean target action |
| `Z_future^σ` generation future visual | `R_t`、`Z_obs`、instruction/text、`A_noise` hidden、`A_cur` generation condition、visual timestep `t_v` | future clean visual target、future 3D GT |
| text/instruction condition | 作为 condition 被读，不被 visual/action tokens 反向改写 | future GT |
| `A_cur` generation condition | 只允许被 `Z_future^σ` 读取 | 不允许被 `R_t`、`Z_obs`、`A_noise` 读取 |

这样 action branch 不会从 future visual 偷答案；但 generation branch 仍能读取
action stream hidden，因此可以学习“这个 action 会导致什么未来视觉变化”。

### Block 内推荐实现

正式版采用 dual-stream / MoT-style implementation：

```text
visual stream:
  Wan2.1 video DiT blocks / visual expert
  tokens: [R_t][Z_obs][Z_future^σ]

action stream:
  action expert / action transformer blocks
  tokens: [A_noise]

cross-stream interaction:
  visual future tokens 可读取 action stream hidden；
  action tokens 可读取 Register/obs/text，但不可读取 future visual；
  可每层交互，也可隔层交互；实现必须满足上表 causal mask。
```

Single hidden-width implementation 只保留为历史 smoke 或 ablation：

```text
[Register][obs][future][action] 全部投影到 D_v
用 attention mask 控制方向
```

它不再是正式结构，因为 action token 被迫使用 video hidden 宽度，policy-only
速度和 Giga/DreamZero-style action decoder 的结构都不够干净。

正式 decode 形式仿照 GigaWorld / DreamZero 的通用做法：

```text
A_noise + condition/context
  -> action stream / action expert hidden
  -> action decoder
  -> predicted action velocity/noise
  -> scheduler / flow denoise
  -> A_out
```

其中 `A_out` 再按数据集 action schema 解码为连续 delta、离散 move/view/stop，
或 VLN simulator action logits / labels。

### Heads

Generation head：

```text
future hidden tokens
  -> visual unpatchify head
  -> predicted velocity/noise for Z_future
  -> [B,16,T_future,H,W]
```

Action/nav head：

```text
action hidden tokens
  -> action decoder
  -> predicted action velocity/noise [B,H_action=10,D_ctrl]
  -> denoise/decode
  -> A_out
```

Stage Three 推理只需要 action stream、必要的 shared/coupled DiT blocks 和
action decoder；generation head 和 video VAE decode 可以不执行。

## 输出怎样解码成视频

视频输出来自 visual stream：

```text
DiT output:
  predicted velocity/noise for future latent
  shape = [B, 16, T_future, H_lat, W_lat]
```

训练时：

```text
loss_visual = MSE(predicted_velocity_or_noise, target_velocity_or_noise)
```

采样/评测时：

```text
Z_future^σ  --denoise steps-->  Z_future
Z_video = concat_or_decode([Z_ref, Z_future])
RGB future consequence = Wan VAE decode(Z_video)
```

第一版只要求 future consequence 质量和运动一致性，不要求恢复完整 dense 81-frame 视频。也就是说，Stage One 的视频监督从“长 dense video generation”转为“给定 obs/action/register 后的 sparse future consequence generation”。这样更接近导航：我们关心动作导致的空间变化，而不是无条件幻想长视频。

## 输出怎样直接转成控制

控制输出来自 action/nav stream。

### 视频数据侧

视频数据没有真实机器人 action 时，先用 pose 构造 pseudo action：

```text
pose_{i}, pose_{i+1}
  -> relative translation / rotation
  -> normalized continuous delta
  -> optional discrete move/view bins
```

Action head 输出：

```text
continuous action:
  predicted action velocity/noise
  denoise/decode 后得到 action delta sequence
  shape = [B, H_action=10, D_ctrl]

or discrete action:
  denoised continuous/embedding action
  -> dataset-specific discrete decoder
  -> action logits or labels
  shape = [B, H_action=10, N_bins]
```

视频侧 action loss 可以是：

```text
continuous:
  action flow/diffusion loss on A_noise -> A_out
  + optional L1/L2 on decoded delta

discrete:
  optional cross entropy on decoded move/view bin
```

Stage One 主线默认仍不启用 action loss；这段定义的是 Stage Three 和后续有可靠
action supervision 时的 action decoder 形式。

### VLN 数据侧

VLN-CE/R2R-CE/RxR-CE 侧直接对应 Habitat 离散控制：

```text
action flow output:
  A_noise [B,H_nav=10,D_ctrl]
    -> predicted action velocity/noise
    -> denoise/decode
    -> nav action chunk [B,H_nav=10]

N_nav_actions:
  STOP
  MOVE_FORWARD
  TURN_LEFT
  TURN_RIGHT
  optional LOOK_UP / LOOK_DOWN
```

推理时：

```text
current RGB + instruction + Register
  -> DiT/action stream with A_noise tokens
  -> action flow decoder for H_nav=10 steps
  -> execute first step or whole 10-step chunk
  -> receive new obs
  -> update Register
  -> repeat
```

推荐先执行 first action，再闭环更新；如果环境步进和模型稳定性允许，再测试 whole
chunk execution。若需要和 Habitat discrete action API 对齐，`A_out` 的最后一步
可以接 discrete decoder / CE auxiliary，但主动作生成形式仍是 noised action
chunk decode。

## 与现有视频数据的吻合方式

已有视频数据主要分两类：

```text
有 pose / sparse pose:
  DL3DV, SpatialVID 等
  可构造 pseudo action 和 future consequence

无可靠 pose:
  只可用于弱视频 consequence 或 VGGT pseudo pose 后再用
```

构造原则不是固定按原始 fps 切 81 帧，而是按空间位移采样：

```text
从 episode 中选当前帧 t
向后搜索 t'
使 pose displacement(t -> t') 落在目标 bucket
```

目标 bucket 要贴近导航尺度：

```text
small:
  ≈ 1 Habitat forward
  ≈ 0.25 m 或数据集归一化后的等价位移

medium:
  ≈ 2-4 Habitat forward
  ≈ 0.5-1.0 m 或等价位移

rotation:
  ≈ 30° / 60° / 90° bins
```

如果数据集没有 metric scale，则使用 episode-level median motion 做归一化，与当前 action annotation specification 的策略保持一致。

## 与导航数据的吻合方式

VLN 数据天然提供：

```text
instruction:
  natural language goal/path instruction

observation:
  RGB frame / panoramic views / simulator observation

action:
  discrete simulator action sequence

time:
  environment step index
```

对齐方式：

```text
每 H_nav=10 个 low-level actions 组成一个 policy chunk
Register 每个 policy chunk 或每次新 observation 更新一次
Stage Three loss 主要监督 action flow / denoised action chunk；必要时附加离散
CE auxiliary 与 Habitat action API 对齐
Stage Two 选中的 hidden/register 层接 3D probe
```

这避免把导航任务硬塞进 81-frame video chunk。`control chunk ≠ Wan 81-frame latent chunk`：导航 chunk 是空间/动作尺度，视频 latent T 只是 consequence supervision 的压缩表示。

## 推荐第一版 tensor 配置

以 896×448、Wan2.1 latent channel 16 为第一版：

```text
Z_obs:
  [B, 16, 1, 56, 112]
  -> 1568 visual tokens

Z_future_noisy:
  Stage One/Two train: [B, 16, 1, 56, 112]
  Stage Three infer:  empty, T=0
  -> 1568 tokens when enabled

Register:
  [B, 128 or 256, D_reg]
  -> projected to DiT hidden
  -> inserted into visual/main stream as history video tokens

Action/Nav tokens:
  video pretrain: [B, 10, D_action]
  VLN policy:    [B, 10, D_action]

Text:
  [B, L_text, D_text]
```

Stage One/Two 训练时典型 token 量：

```text
visual/main tokens = 128/256 history + 1568 obs + 1568 future
action tokens = 10
text tokens = prompt length
```

Stage Three policy-only 推理时：

```text
visual tokens = 1568 obs only
register tokens = 128/256
action tokens = 10
future visual tokens = 0
```

如果后续将分辨率降到 448×448：

```text
H_lat × W_lat = 56 × 56
1 latent time visual tokens = 784
Stage Three visual tokens = 784
```

这会更接近 GigaWorld 的 token 量，但需要先确认 Wan2.1/InfiniteWorld 的预训练和推理脚本在该分辨率下稳定。

## 训练阶段对应关系

### Stage One：视频 + pseudo action

目标：

```text
obs/register/instruction/action -> future visual consequence
obs/register/instruction -> action pseudo label
```

损失：

```text
L_stage1 = L_visual_diffusion + λ_action L_action
```

future visual 可以读 action，action 不读 future visual。

### Stage Two：视频生成方式中的 3D 监督

模型结构不大改，只加 3D probe：

```text
selected hidden/register layer
  -> depth / pose / point / occupancy / correspondence probe
```

损失：

```text
L_stage2 = L_stage1 + λ_3d L_3d
```

3D GT 来自真实 pose / sparse pose interpolation / VGGT pseudo labels。future 3D 同样是 consequence supervision，不作为 action condition。

### Stage Three：VLN policy

删除 future visual branch：

```text
obs/register/instruction
  -> A_noise action tokens
  -> action velocity/noise
  -> denoise/decode
  -> control
```

损失：

```text
L_stage3 = L_action_flow(A_noise -> A_out, GT actions)
         + optional auxiliary 3D regularizer
         + optional discrete CE / action consistency loss
```

## 当前未实现与必须验证项

1. 需在代码中确认 Wan2.1 / InfiniteWorld hidden_dim、patch_embed、text cross-attention 接口，再定 `D_reg` 和 `D_action`。
2. 需实现 history/Register video tokens 的插入、读回和更新，避免 B 方案只把
   Register 当静态 text condition。
3. 需做 `T_future=1` 的视频 consequence smoke，确认 Wan VAE decode 的 sparse frame pack 合理。
4. 需比较 `H_nav=10` 下 first-action closed-loop 与 whole-chunk execution 的闭环导航效果。
5. 需做不同 displacement bucket 的 ablation，确认视频 pseudo action 与 VLN control scale 对齐。

## 待定问题与当前选择

完整登记表见 `NAV-DES-005`：

```text
01_design/v1_open_design_questions.md
```

本文件只定义新版 I/O 接口；所有待定参数、当前默认选择、选择理由和冻结条件统一
维护在 `NAV-DES-005`，避免多处重复后出现口径漂移。
