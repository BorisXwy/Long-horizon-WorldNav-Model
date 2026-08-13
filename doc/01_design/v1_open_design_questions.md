# V1 NAV 新版模型待定项与当前选择

| 字段 | 内容 |
| --- | --- |
| ID | NAV-DES-005 |
| 类型 | 设计待定项登记表 |
| 状态 | Live / Open Questions |
| 更新时间 | 2026-08-14 |
| 职责 | 集中记录新版 GigaWorld-like + Register NAV 方案中尚未冻结的结构、参数、当前默认选择、选择理由和验证条件 |

## 当前总览

新版模型的大方向已经确定：

```text
history/Register 作为 visual/main stream video tokens
current observation 作为 clean visual tokens
future visual 作为 generation branch 的 noised consequence tokens
action/nav 作为独立 action stream tokens，是待生成变量，不是 policy condition
text/timestep/modality/action-scale 作为 condition
Stage Three 推理删除 generation branch，只保留 policy-only 路径
```

但仍有若干参数和结构必须通过 smoke、ablation 或代码兼容性检查冻结。本文是当前
实现前的待定项登记表；若后续实验确认，应同步更新 `decision_log.md` 和对应
训练/代码配置。

## 状态定义

```text
Accepted Design / Not Implemented:
  设计选择已经采纳，但代码尚未实现或未完全验证。

Proposed:
  当前推荐默认值，允许实验后修改。

Open:
  尚未有足够依据定默认值，需要先做实验或查代码。

Ablation:
  不作为主方案，但应保留对照价值。
```

## 待定项总表

| 编号 | 问题 | 当前选择 | 状态 | 选择理由 | 冻结条件 |
| --- | --- | --- | --- | --- | --- |
| Q01 | history/Register 放在哪里 | 默认放在 visual/main stream，作为 video tokens；B-style text condition Register 只做 ablation | Accepted Design / Not Implemented | history 是时空状态，不应只是 cross-attention 静态提示；放进 video stream 才能产生 `Register-after-DiT` | 实现 token packing、读回 Register state，并完成一次 forward smoke |
| Q02 | generation branch 的 `Z_future T=1` 是否只生成一帧 | 否。`T=1` 是 one future latent time cell；若使用 `[obs, future_1..future_4]` sparse pack，decode 后约为 1+4 RGB frames | Accepted Design / Not Implemented | Wan VAE temporal factor≈4；`T_lat=2` 可由 5 RGB frames 得到，其中第一个 latent 是 obs，第二个 latent 是 future consequence | 完成 VAE encode/decode smoke，确认 sparse pack 视觉合理 |
| Q03 | generation branch 的命名 | 统一称 `generation branch` 或 `future consequence branch` | Accepted Naming | 避免继续用“视频分支”误导为 dense long video generation | 文档、config、代码变量命名同步 |
| Q04 | Register token 数 `K` | 第一版候选 `K=128`，对照 `K=256` | Open | 128 更快，256 可能保留更多几何/历史信息 | 比较显存、latency、Stage One loss、Stage Two 3D probe |
| Q05 | Register 维度 `D_reg` | 先低维表示，再投影到 Wan2.1 DiT hidden_dim | Open | 直接 full hidden 可能贵；低维可降低 updater 成本 | 读取本地 Wan2.1/InfiniteWorld config，确认 hidden_dim 与 projection 位置 |
| Q06 | `R_0` 初始方式 | `R_{-1}=R_null` fixed template；`R_0=RegisterCell(R_null, concat([visual_tokens(C_0), A_hist0]))`；无历史动作时用 no-op/empty `A_hist0` | Accepted / Implemented in full-pipeline smoke | 避免 learnable initial_register 藏场景先验；首步和后续都走同一个 cell，结构更统一 | `NAV/scripts/run_v1_full_pipeline_smoke.sh` 已验证 RegisterCell roll/update |
| Q07 | RegisterCell 是否读 previous action/motion | 读，但只读独立 `A_hist` tokens / cross-attention context；禁止 action bias / latent bias；不再区分 Extractor/Updater | Accepted Design / Not Implemented | 历史动作有助于解释历史视觉变化；统一 RegisterCell 减少首步/后续结构差异；action 不以 additive bias 污染视觉 latent/Register | 代码接口检查：RegisterCell 接受 `R_prev, visual_tokens, A_hist_tokens`，无 additive bias 路径；mask 单元测试 |
| Q08 | video stream token packing 顺序 | 倾向 `[Register][obs][future]` | Proposed | Register prefix 最像 memory，便于读回；obs 在 future 前保留 clean context | 检查 patch/position embedding 兼容性；forward smoke |
| Q09 | policy Register 是否读 future | 不读 future；future branch 可读 Register/action/obs | Accepted Design / Not Implemented | 防止训练时 future leakage 污染 Stage Three policy memory | attention mask 单元测试；确认 policy hidden 对 future token mask 为 false |
| Q10 | `T_future` 第一版取值 | `T_future=1` | Accepted Design / Not Implemented | 最大限度降低 generation branch token；仍保留动作导致视觉变化的监督 | sparse future consequence smoke；必要时对照 `T_future=2/4` |
| Q11 | 输入分辨率 | 第一版 448×896 保持 InfiniteWorld 兼容；随后测 448×448 或 384×320 | Proposed | 448×896 可复用旧流程；低分辨率更接近 GigaWorld token 量 | 代码兼容性、速度、显存、VBench/导航效果 |
| Q12 | 导航 horizon `H_nav` | 固定 `H_nav=10`；闭环默认只执行 first action | Accepted Design / Not Implemented | 统一 Stage Three action chunk 形状；保留局部轨迹意图，同时不强迫 10 步开环执行 | 实现 action flow decoder 后做 first-action vs whole-chunk 闭环 ablation |
| Q13 | 视频 action horizon `H_a` | 固定 `H_action=10`，视频 pseudo motion 重采样到 10 slots | Accepted Design / Not Implemented | 与 VLN action horizon 对齐，避免 Stage One/Two/Three 更换 action token 数 | 完成 action schema 与 dataloader smoke，统计重采样后的 motion 分布 |
| Q14 | action 输出形式 | `A_noise -> action flow decoder -> A_out`；离散 CE 只作为 auxiliary / decode 对齐 | Accepted Design / Not Implemented | 对齐 GigaWorld/DreamZero 的通用 action decode 方式；避免 direct CE head 变成旁路小结构 | action flow loss smoke、policy-only latency、VLN SR/SPL |
| Q15 | 视频 pseudo action 尺度 | 有 metric pose 用 meter/radian；无 metric scale 用 episode-normalized displacement bins | Proposed | 多数据集尺度不一致，需要归一化后才能混训 | 数据集间 action 分布统计；混训 loss 稳定性 |
| Q16 | Stage Two probe 层 | 候选 25% / 50% / 75% / final layer | Open | early/middle/late 层可能分别偏几何、运动、语义/生成 | depth/pose/correspondence/occupancy probe 实验 |
| Q17 | Register 是否 latent-like spatial shape | 主方案先用 compact tokens；latent-like register 只做可选对照 | Proposed | compact tokens 更适合 policy-only；latent-like 可能更利于 generation 但更贵 | compact vs latent-like 生成指标和导航指标 |
| Q18 | action stream 与 video stream 的交互层数 | 倾向每层或大部分层交互，但用 mask 控制方向 | Open | 交互太少 cotrain 弱，交互太多成本高且 leakage 风险大 | latency、显存、mask 单元测试、loss 曲线 |
| Q19 | Stage Three 是否保留 generation branch 辅助 | 推理删除；训练可保留 optional auxiliary，但不得作为 policy condition | Accepted Design / Not Implemented | 保留 shared representation，推理保持小输入 | Stage Three with/without auxiliary 对照 |
| Q20 | VLN observation 形式 | 第一版 single RGB 或固定 layout；panorama/multiview 后续对照 | Open | single RGB 最简单，panorama 更符合 VLN 但会增加 token | R2R-CE/RxR-CE 数据接口与速度对照 |
| Q21 | 无 pose / 无 action 视频如何使用 | 只作为弱 generation/caption 数据，或经 VGGT/flow/inverse dynamics 产生低置信 pseudo motion 后使用；不作为主 Register history 训练 | Proposed | 通用视频和生成 benchmark 常无 pose/action，主方法不能依赖不可获得标注 | confidence 分布审计；有/无 pseudo motion 混训 ablation |
| Q22 | pose 在 V1 中的角色 | pose 是训练 teacher、pseudo action 生成和 3D supervision 来源；不是模型 condition，也不是 Stage Three 推理输入 | Accepted Design / Not Implemented | 避免把不可获得的 GT pose 作为隐含依赖，保证真实世界可部署性 | 代码接口检查：forward 不接受 GT pose 作为 policy condition |
| Q23 | action 在 V1 中的角色 | current/target action 在 policy 侧不是输入；`A_noise` 是 action stream 的 noised generated variable。Generation branch 可读取 action stream hidden 来生成 future consequence | Accepted Design / Not Implemented | 避免 action 同时作为输入和输出的语义冲突；保留 video generation 对 action 的条件依赖 | forward 接口检查；mask 检查；action ablation |
| Q24 | DiT 内部采用单流还是双流实现 | 正式版采用 dual-stream / MoT-style visual stream + action expert；single hidden-width 只作历史 smoke/ablation | Accepted Design / Not Implemented | 两个 branch 的 raw input、token 数、输出目标差异很大；双流更接近 Giga/DreamZero/Fast-WAM 的 policy latency 路线 | dual-stream forward smoke、causal mask 单元测试、policy-only latency |

## 当前建议默认实现

第一版最小可实现版本建议固定为：

```text
Register:
  K = 128
  R_{-1} = fixed R_null
  R_t = RegisterCell(R_{t-1}, concat([visual_tokens(C_t), A_hist_t]))
  inserted as visual/main stream prefix tokens

Visual:
  resolution = 448 × 896 initially
  Z_obs      = [B,16,1,56,112]
  Z_future   = [B,16,1,56,112] during Stage One/Two
  Z_future   = empty during Stage Three inference

Action/Nav:
  H_action = H_nav = 10
  action tokens shared by video pretraining and VLN, but treated as A_noise generated variables
  action decoder uses Giga/DreamZero-style flow/diffusion decode

Mask:
  action/nav tokens cannot read future visual/3D
  policy Register cannot read future visual/3D
  generation branch can read obs/Register/action/text

DiT implementation:
  formal:
    dual-stream / MoT-style visual stream + action expert stream
  ablation/history only:
    single hidden-width tokens + explicit attention mask

Training:
  Stage One:
    video pseudo action + sparse future consequence
  Stage Two:
    same generation-style forward + 3D probe/loss
  Stage Three:
    policy-only path + optional auxiliary, no future condition
```

## 当前最优先要验证的五件事

1. `Z_future T=1` sparse pack 经 Wan VAE decode 是否有合理视频 consequence。
2. `[Register][obs][future]` packing 是否能与现有 Wan2.1/InfiniteWorld patch/position embedding 兼容。
3. attention mask 是否能严格保证 policy Register/action 不读 future。
4. `H_nav=10` 下 first-action closed-loop 与 whole-chunk execution 哪个更稳定。
5. `K=128` 是否足够支撑 Stage Two 3D probe；若不足再升到 `K=256`。
