# NAV 关键决策日志

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-002` |
| 类型 | 决策日志（Decision Log） |
| 状态 | Live |
| 更新时间 | 2026-08-14 |
| 职责 | 记录会影响模型、数据、训练或评测口径的已确认决策 |

## 决策表

| ID | 日期 | 状态 | 决策 |
| --- | --- | --- | --- |
| DEC-001 | 2026-07-24 | Accepted | A/B 均从原始 InfiniteWorld 权重独立初始化，不互相继承 |
| DEC-002 | 2026-07-24 | Accepted | 保留 InfiniteWorld local memory；删除 HPMC history，并分别以 latent prefix 或 DiT condition 注入 Register |
| DEC-003 | 2026-07-27 | Accepted | 训练采用 Short 2–3、Medium 4–7、Long ≥8 chunks 的课程 |
| DEC-004 | 2026-07-27 | Accepted | 原始下载目录只读；manifest 与 latent 统一写入 `/sharedata/NAV/derived/` |
| DEC-005 | 2026-07-27 | Accepted | latent 使用原位可扩展缓存，后续阶段只补新增 chunks |
| DEC-006 | 2026-07-27 | Accepted | Kinetics VGGT 标注暂不进入本轮 NAV 训练 |
| DEC-007 | 2026-07-25 | Accepted | 结果统一写入 `NAV/result/<evaluation>/<experiment>/` |
| DEC-008 | 2026-07-28 | Accepted | 优先从 A/B 各自 RE10K 1000-step 全参 checkpoint 继续 SpatialVID 2–3 chunk short-history；latent/action 并行离线准备，Register 在 forward 内以 teacher forcing 在线递归计算 |
| DEC-009 | 2026-07-29 | Accepted | DL3DV Pose 只与官方 `images_8` 抽帧一一对齐；不再把Pose序号解释为原MP4帧号，新cache隔离写入 `latents_pose_aligned` |
| DEC-010 | 2026-07-29 | Accepted | 当前训练数据只缓存RGB/VAE latent与Action，Register一律在forward内在线计算；Short采用2–3 chunk random-prefix last-target teacher forcing |
| DEC-011 | 2026-07-29 | Accepted | SpatialVID VAE缓存改为7个稳定shard并行：GPU 0三个、GPU 1四个；以 `index mod 7` 隔离写入并保留原子续传 |
| DEC-012 | 2026-07-30 | Proposed / Semantics Accepted | Stage Two 完全保留视频生成前向与Diffusion/RFlow目标，只增加多层Register-after-DiT probe和选定层3D监督；Stage Two不包含导航任务 |
| DEC-013 | 2026-07-30 | Accepted Design / Not Implemented | Stage One改为新episode首个History Chunk经Extractor产生Register，后续由Updater递归更新；checkpoint不保存episode-specific Register，也不采用携带场景先验的learnable initial state |
| DEC-014 | 2026-07-30 | Proposed / Semantics Accepted | Stage Three才引入导航任务，读取Stage Two probe选定层的Register-after-DiT表示；导航History只来自真实Observation，generated future不得回灌 |
| DEC-015 | 2026-07-30 | Accepted | Stage One 1.0从原始Infinite-World初始化，只用DL3DV，全参数训练；1/2/3-history等比例shuffle采样，首Chunk用Extractor、后续用Updater，A/B均不加载旧RE10K checkpoint |
| DEC-016 | 2026-07-30 | Measured / Waiting | GPU 1实测A BS4 OOM、A BS1峰值29.21GiB、B BS1峰值16.48GiB；A已超过半卡，暂不冒险启动A/B同卡并发1000-step，等待GPU分配决策 |
| DEC-017 | 2026-07-30 | Running | Stage One 1.0改为GPU0训练A、GPU1训练B；GPU0保留1个latent worker。启动时GPU1被其他账号占用约41GiB，B进入45,000MiB free-memory等待队列，启动并完成首步后自动增加3个GPU1 latent workers |
| DEC-018 | 2026-07-31 | Accepted | Stage One 1.0正式训练从1000扩展为10000 optimizer steps，每1000 steps保存完整checkpoint；旧A-1000在step373停止并作为历史run保留，新A/B-10000均从原始Infinite-World重新初始化 |
| DEC-019 | 2026-08-03 | Accepted | Stage One 1.0正式训练改为1000 effective optimizer steps、effective batch size 16（A: micro=1×accum=16；B: micro=2×accum=8），每100 effective steps保存完整checkpoint；A/B均from scratch从原始Infinite-World初始化，不加载旧RE10K或旧A/B-10000 checkpoint；其余显存继续并行准备full_episodes_v1 latent |
| DEC-020 | 2026-08-07 | Accepted / In Progress | 闭环完整训练主数据固定为 R2R-CE、RxR-CE、LHPR-VLN、ScaleVLN 四套，统一落盘 `/sharedata/datasets/{R2R,RxR,LHPR-VLN,ScaleVLN}`；以 CE/任务标注为训练入口，原始 RxR GCS pose traces 与 StreamVLN CE 子集为可选增强 |
| DEC-021 | 2026-08-07 | Accepted / In Progress | R2R-CE/RxR-CE 外部 SOTA 复现只纳入文献 R2R Val-Unseen SR≥StreamVLN（含 StreamVLN）且有公开权重的工作；仓库 clone 到 `3d_wm_vln/`，环境用 `virtual_env/.venv_*` venv；协议见 NAV-EVL-002 |
| DEC-022 | 2026-08-08 | Proposed / Working Hypothesis | Stage Three：生成+Register 改 NavPolicy 时，训练侧按接近视频 chunk 的**空间变化尺度**更新 Register（观测常仅 ~4 frame）；推理快则可单步决策，慢则与 Register 更新同步成半闭环。控制 chunk≠81 帧 VAE chunk。详见 NAV-DES-003、insight R17 |
| DEC-023 | 2026-08-09 | Accepted Design / Not Implemented | NAV 采用 action-centered causal interface：future visual / future 3D 只作为动作和Instruction/Goal条件下的 consequence supervision，不作为 policy condition；Stage One/Two/Three 共享同一 DiT/Register backbone，Stage Three 使用其 policy-only裁剪路径而非新建 perception backbone |
| DEC-024 | 2026-08-09 | Accepted Design / Not Implemented | 新版 NAV 中 history/Register 默认作为 visual/main stream 的 video tokens 进入 DiT，而不是作为 text/condition 侧静态 token；B-style condition Register 只保留为 ablation。详见 NAV-DES-004 |
| DEC-025 | 2026-08-10 | Accepted Design / Not Implemented | V1 中 current/target action 不作为 policy condition；action/nav tokens 是 policy 要生成的 query/noised variables。Generation branch 可读取 DiT 内 action stream hidden 来生成 future consequence；history 侧只可使用 previous executed action。详见 NAV-DES-004/005 与 NAV-TRN-006 |
| DEC-026 | 2026-08-10 | Accepted / Smoke Verified | V1 Stage One generation horizon 固定为 `T_latent=4`：每个 micro chunk 使用13帧、stride 12独立 VAE encode；与 InfiniteWorld 的对齐只对齐 history/context span，IW-1/4/8/16 chunks 分别换算为7/27/54/107个 micro history steps；target loss 仍只算 T4 short future。详见 NAV-TRN-007 |
| DEC-034 | 2026-08-14 | Accepted Design / Not Implemented | V1 action path 从 `A_query` 正式改为 `A_noise`；`H_action=H_nav=10`；动作输出仿照 GigaWorld/DreamZero 的 action flow/diffusion decode；backbone 改为 dual-stream / MoT-style；causal attention 关系冻结。详见 NAV-DES-004/005、NAV-TRN-006 |
| DEC-035 | 2026-08-14 | Accepted Design / Not Implemented | Register Extractor / Updater 不再读取 `A_hist`、pose、odometry 或 pseudo motion；Register 只由历史/current visual observation 更新。`A_hist` 若保留，只能作为 Register 外部 action/state context、训练监督或审计字段。 |
| DEC-036 | 2026-08-14 | Accepted Design / Not Implemented | 覆盖 DEC-035：`A_hist` 重新参与 Register Extractor / Updater，但只能以独立 action tokens / cross-attention context 交互；禁止 action bias / latent bias / additive bias 方案。 |
| DEC-037 | 2026-08-14 | Accepted Design / Not Implemented | 覆盖 DEC-036 的双模块表述：Register 侧统一为 `RegisterCell`。每个 episode 从 fixed `R_null` 开始，首步和后续都执行 `R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`；继续禁止 action/latent/additive bias。 |

## 记录要求

### DEC-022 补充（2026-08-08）

- **状态**：`Proposed`（工作设想登记，未实现、未冻结 packing/损失）。
- **选择**：Nav 适配时 Register 更新对齐「chunk 级空间位移」而非强凑 81 RGB；推理频率可按算力在单步与 Register 同步之间切换。
- **理由**：视频帧间微动 vs Habitat 大步进的数量级对照表明，~4 Habitat step 的位移接近 1 个室内 video chunk，适合半闭环局部原语；与 DEC-014（真实 Observation History）不冲突。
- **影响文档**：`01_design/v0_stage_three_navigation.md`（NAV-DES-003）、`08_insight/insight_log.md`（R17）。
- **替代关系**：不替代 DEC-014；细化 Stage Three 控制频率，待实验后再升为 Accepted。

### DEC-023 补充（2026-08-09）

- **状态**：`Accepted Design / Not Implemented`（设计原则已写入
  `NAV-DES-001/002/003`，具体 attention mask、token slicing 和训练代码尚未实现）。
- **选择**：采用 action-centered / navigation-centered 因果方向：
  `history/current obs/instruction → action/nav decision → future visual/3D consequence`。
  Future visual / future 3D 可作为 dense auxiliary supervision，但不得作为
  action/navigation token 的输入条件。
- **理由**：一个当前观测在没有 action 和 goal/text 时对应多种未来；让 action
  读取生成未来会引入不适定 future leakage，并放大 video token 带来的训推成本。
  Fast-WAM 与 GigaWorld-Policy 的共同启发是：训练时保留 future/world branch
  塑造 shared representation，推理时裁剪为 action-only / policy-only 小输入路径。
- **影响文档**：`01_design/v0_stage_one_register_world_model.md`、
  `01_design/v0_stage_two_3d_supervision.md`、
  `01_design/v0_stage_three_navigation.md`。
- **替代关系**：不替代 DEC-014；进一步明确“generated future 不回灌 History”
  之外，future visual/3D 也不作为 policy condition。

### DEC-024 补充（2026-08-09）

- **状态**：`Accepted Design / Not Implemented`（接口设计已写入
  `NAV-DES-004`，代码中的 token packing 和可读回 Register state 尚未实现）。
- **选择**：history/Register 默认放在 visual/main stream，作为 video tokens 与
  current observation、noisy future latent 共同参与 DiT self-attention / MoT
  计算；不默认拼到 text condition 侧。
- **理由**：history 是时空状态，不应只是 cross-attention 里的静态提示。放在
  video/main stream 后，Register 能与 observation/future latent 在同一套时空
  backbone 中交互，也更便于 Stage Two probe 和 Stage Three policy 读取
  `Register-after-DiT`。
- **影响文档**：`01_design/v1_action_centered_io_interface.md`。
- **替代关系**：细化并部分替代 DEC-002 中 B 方案的默认地位；B-style
  condition Register 仍可作为 ablation，不作为新版主方案。

### DEC-025 补充（2026-08-10）

- **状态**：`Accepted Design / Not Implemented`（语义已写入
  `NAV-DES-004/005` 与 `NAV-TRN-006`，代码接口和 mask 尚未实现）。
- **选择**：当前/目标 action 不作为 policy condition。Policy 侧的
  action/nav tokens 是 query 或 noised target，是模型要生成/去噪/分类的变量。
  Generation branch 可以读取 DiT backbone 内的 action stream hidden state，
  用于预测该 action 导致的 future visual consequence。
- **理由**：若 clean target action 同时作为输入和输出，会造成 policy 侧语义
  泄漏；但 generation branch 又必须知道 action，才能学习 action-stream-conditioned
  future consequence。把 action 放在 DiT 内部 action stream，既避免 policy
  leakage，又保留 video generation 所需的 action-stream 条件。
- **影响文档**：`01_design/v1_action_centered_io_interface.md`、
  `01_design/v1_open_design_questions.md`、
  `04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：细化 DEC-023 的 action-centered interface；不改变
  DEC-024 中 history/Register 作为 visual/main stream token 的选择。

### DEC-026 补充（2026-08-10）

- **状态**：`Accepted / Smoke Verified`（manifest、T4 VAE encode smoke 和
  1-step training smoke 已完成；全量 T4 latent 与正式训练尚未完成）。
- **选择**：V1 Stage One 的 generation branch 默认 `target_latent_t=4`，
  对应每次从13帧 RGB 独立编码得到 `[B,16,4,H,W]` target；相邻 micro chunk
  使用 stride 12，共享1帧边界。
- **理由**：保留小 noisy target 带来的训练/推理速度优势；长程能力通过
  Register 覆盖的真实 context span 对齐 InfiniteWorld，而不是把 target
  恢复成81帧 dense video。
- **影响文档**：`04_training/v1_stage_one_t4_iw_aligned.md`、
  `03_data/training_data_construction.md`、
  `04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：替代 V1 早期 sparse pack 中 `Z_future T=1` 作为主线训练
  horizon 的临时选择；不删除 T=1 作为速度 ablation 的可能性。

### DEC-027 补充（2026-08-11）

- **状态**：`Accepted Design / Not Implemented`（三阶段训练目标已写入
  `NAV-DES-004` 与 `NAV-TRN-006`；最终 Stage One action interface 代码尚未实现）。
- **选择**：Stage One 不把视频 pseudo action 训练成 policy。Stage One 的主目标
  是 video generation memory pretraining：训练 Register long-history update 与
  future visual consequence generation。Action 在 Stage One 只保留三类接口语义：
  1) history action / previous motion 可选参与 Register update；
  2) current action 可选作为 video generation condition；
  3) noised/query action tokens 与 action output head 保留格式，但默认不计算
  policy/action loss。真正的 action/nav output 在 Stage Three 用 VLN/R2R/RxR
  action label 训练。
- **理由**：视频生成数据通常没有 instruction，也没有真实 embodied policy action；
  强行用 pseudo motion 训练 action output 会让 Stage One 的 action 语义偏离
  Stage Three 导航任务。保留接口格式可以保证 backbone/action branch 后续可接，
  但避免把无 policy 意义的监督写进最终能力假设。
- **影响文档**：`01_design/v1_action_centered_io_interface.md`、
  `04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：细化并部分替代 DEC-025：保留 action-centered 因果接口，但将
  Stage One 的 action loss 从默认训练目标降级为 format/interface optional；
  Stage Three 才是 policy/action supervision 主阶段。

### DEC-028 补充（2026-08-11）

- **状态**：`Accepted Design / Not Implemented`（Stage Three replay/cotrain
  方案已写入 `NAV-TRN-006`；具体 multi-task dataloader 和 loss mixing 尚未实现）。
- **选择**：Stage Three 不能只做 VLN policy-only fine-tune。主任务仍是
  VLN action/nav supervision，但训练中需要 replay Stage One video generation
  windows 与 Stage Two 3D supervision windows，组成 cotrain：
  `L_stage3 = L_nav + λ_replay_video L_visual_flow_replay + λ_replay_3d L_3D_replay`。
  推理仍然 policy-only，generation/3D heads 不执行。
- **理由**：完全 policy-only fine-tune 可能破坏 Stage One 的长历史视频记忆和
  Stage Two 的空间表征，使 shared backbone/Register 退化。Replay/cotrain 作为
  regularization，保留世界模型和 3D 能力，同时让 action head 适配 VLN。
- **影响文档**：`04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：替代 Stage Three 第一版“推理删除 generation，训练也不构建
  auxiliary”的简化设定；保留 S3-R0 policy-only 作为 baseline/ablation。

### DEC-029 补充（2026-08-11）

- **状态**：`Accepted Design / Not Implemented`（Stage Two additional-loss 方案已写入
  `NAV-TRN-006`）。
- **选择**：Stage Two 不叫 cotrain，也不默认引入单独 Stage One replay batch。
  Stage Two 是在 Stage One 的同一个 video generation forward/loss 上直接增加
  3D supervision/probe：`L_stage2 = L_stage1 + λ_3d L_3D`。
- **理由**：Stage Two 的语义是“视频生成前向中加入空间监督”，而不是多任务数据
  混训。这样命名更准确，也避免实现时误以为需要独立的 Stage One replay sampler。
  由于 `L_stage1` 始终存在，generation/Register memory 本身已经在同一 batch 中
  被持续约束。
- **影响文档**：`04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：细化 Stage Two 的“same generation-style forward + 3D probe”
  设定；替代“Stage Two cotrain/replay”的错误命名。

### DEC-030 补充（2026-08-11）

- **状态**：`Superseded / Historical Implementation Removed`（旧
  InfiniteRegisterAdapter/Wan T4 训练入口已在 2026-08-14 从当前代码删除；本决策
  只保留“V1 从官方 Wan2.1-T2V-1.3B 初始化”的原则）。
- **选择**：V1 Stage One 正式训练从官方 `Wan2.1-T2V-1.3B`
  `diffusion_pytorch_model.safetensors` 初始化 DiT backbone，而不是从
  InfiniteWorld checkpoint 初始化。官方 Wan 的 `patch_embedding.weight`
  为16 latent channels；V1 正式结构保持同样的 `16 -> 1536`
  patch stem，因此官方权重可以 shape-match 直接加载。
- **理由**：新版方案目标是基于 Wan2.1-1.3B backbone 建立近似 GigaWorld/DreamZero
  的 shared DiT/Register/action interface；继续从 InfiniteWorld checkpoint
  初始化会混入 HPMC/action/local memory 旧方案先验，不符合“正式 Stage One”
  的 from-scratch 定义。
- **影响文档**：`00_overview/decision_log.md`；旧影响代码已删除，当前实现见
  `src/nav/v1/models/full_model.py` 和 `NAV-EVL-005`。
- **替代关系**：替代旧 V0/V1.0 实验中的“从原始 InfiniteWorld 初始化”规则。
  InfiniteWorld checkpoint 仍只作为 V0 对照、旧实验复现和 baseline 使用。

### DEC-031 补充（2026-08-13）

- **状态**：`Superseded by DEC-034/037/038`（旧 `A_query` 实现已删除；主线改为
  `A_noise` + dual-stream + `RegisterCell`）。
- **选择**：正式 V1 Stage One 必须把 `A_cur` 作为 DiT condition/context；
  `A_query/A_noise` 才作为 shared action tokens 插入 Wan/DiT 主 token stream。
  `A_out` 从同一 Wan block 后的 shared action hidden 读出。Stage One 只计算
  video diffusion / flow loss，`lambda_action=0`，但 `A_query` 仍通过 future
  video loss 的 shared attention 路径接收梯度。
- **实现口径**：Wan self-attention 对真实 video tokens 应用 RoPE；`A_query`
  action tokens 无 HxW 空间位置，不应用 RoPE，但参与每层 self-attention、
  text cross-attention 和 FFN。`A_cur` 是 generation-only condition，只通过
  noisy-only cross-attention 作用到 future visual tokens，不改写 history prefix
  或 `A_query` tokens。policy-only 前向不输入 future noisy video，也不输入 current
  action condition，只输入 Register/local/text +
  `A_query` tokens。为避免 action 双路注入，正式 V1 中原 InfiniteWorld
  `action_encoder` 接收 no-op；`A_cur` 只通过 DiT condition/context 约束
  video generation branch。
- **废弃内容**：`Register/local -> 旁路小 action head` 是错误 scaffold，不得再作为
  正式 Stage One、policy/video shared backbone 或 policy latency 结论。旧脚本
  `benchmark_v1_policy_like_latency.py` 只保留为历史旁路测速记录。
- **验证记录**：
  - 1-step training smoke：
    `NAV/log/v1-final-shared-action-token-smoke-20260813/full-step-000001.pt`
    已包含 `shared_action` 参数组。
  - shared policy latency：
    `NAV/result/latency/v1_final_shared_policy_step1000_20260813.json`；
    单次 policy-only full shared Wan/DiT forward 约 `0.366s`，峰值显存约 `5.78GiB`。
- **影响代码**：旧影响代码已从当前 NAV 主线删除；当前结构验证见
  `NAV/src/nav/v1/models/full_model.py`、`NAV/scripts/smoke_v1_full_pipeline.py`。
- **替代关系**：替代 DEC-027/028/030 中“action branch 保留格式但未明确必须进入
  shared backbone”的含混表述。

### DEC-032 补充（2026-08-13）

- **状态**：`Superseded / Historical`（16-channel/token-type 原则保留；旧 Wan
  patch 入口脚本已删除）。
- **选择**：正式 V1 删除 InfiniteWorld 的 `20-channel mask` 输入包装。DiT 主输入
  不再做 `16 latent channels + 4 condition mask channels` 的 channel concat；
  `patch_embedding` 回到 Wan 原生 `Conv3d(16 -> 1536, kernel=(1,2,2))`。
- **替代机制**：在 patch embedding 之后加入 `token_type_embedding`，显式标记
  `prefix/history/local`、`future noisy visual`、`A_query/action` 三类 token。
  继续保留 `num_c / prefix-noisy attention split`，用于保证 clean prefix /
  `A_query` 不被 future noisy visual 反向污染。
- **输出语义**：video denoise head 只作用在 future noisy visual tokens 上，输出
  `Z_future` 的 velocity / flow prediction；clean prefix/history/local tokens
  只作为 context，不再进入 video reconstruction head。
- **验证记录**：
  - 微型 DiT forward smoke：`in_channels=16`、`token_type_embedding=(3,D)`、
    video 输出 `T_future`、policy-only 输出 `shared_action_hidden`，均通过。
  - 正式 1.3B 官方 Wan 加载审计：`patch_embedding.weight=(1536,16,1,2,2)`，
    `loaded_keys=825`，`mismatched_keys=0`，无 20-channel partial load。
  - 1-step 正式训练 smoke：
    `NAV/log/v1-final-16ch-token-type-acur-noisyonly-smoke-20260813/full-step-000001.pt`；
    `loss=1.1650`，`peak_reserved_gib=31.17`。
- **影响代码**：旧影响代码已从当前 NAV 主线删除；当前 full-pipeline smoke
  用 `V1FullWorldNavModel` 验证 no 20-channel mask / no action-bias 结构约束。
- **替代关系**：替代 DEC-030 中“20-channel IW-style backbone 入口 + 4 mask
  channels zero-init”的实现口径；旧 V0/IW baseline 脚本仍可作为历史复现实验，
  但不得作为 V1 正式结构。

### DEC-033 补充（2026-08-13）

- **状态**：`Accepted / Implemented / Smoke Verified`。
- **选择**：正式 V1 采用 GigaWorld-Policy-style per-token timestep：
  clean prefix / Register / local visual tokens 使用 `t=0`；future noisy visual
  tokens 使用 RFlow `visual_t`；`A_query/A_noise` action tokens 使用独立
  `action_t`。
- **训练实现（历史）**：旧 Wan T4 脚本曾显式采样 `visual_t` 并传给
  RFlowScheduler 加噪和 video flow loss；同时用 `action_flow_shift=5.0`
  独立采样 `action_t`，传给 backbone 只调制 `A_query` action tokens。
  当前 visual branch 继续使用 `visual_flow_shift=7.0`、
  `use_timestep_transform=True`、`use_reversed_velocity=True`，保持 Stage One
  video loss 口径不变。
- **DiT 内部语义**：Wan block 的 AdaLN / modulation 拆成三路：
  prefix 使用 `e_no_noise`，action prefix 使用 `e_action`，future noisy 使用
  `e_visual`。full self-attention 中 future visual 可以读取 prefix/action；
  prefix/action 不被 future visual 反向更新。
- **loss 语义**：Stage One 仍只计算 `L_visual_flow`，`lambda_action=0`；
  action token 的 `action_t` 不是动作监督，而是为 Stage Three policy/action
  denoise 路径提前固定输入格式。
- **验证记录**：
  - 微型 DiT forward smoke：显式传 `visual_t=777`、`action_t=333`，video /
    action hidden / policy-only 均通过。
  - 正式 1-step training smoke：
    `NAV/log/v1-final-giga-timestep-smoke-20260813/full-step-000001.pt`；
    `loss=1.1692`，`visual_timestep_mean=917.26`，
    `action_timestep_mean=998.0`，`peak_reserved_gib=30.95`。
- **影响代码**：旧影响代码已从当前 NAV 主线删除；当前 full-pipeline smoke
  仍保留 `visual_t` 与独立 `action_t` 两路输入格式。
- **替代关系**：替代 DEC-032 smoke 版本中 `A_query` action tokens 默认使用
  `t=0` 的实现；DEC-032 的 16-channel/token-type/20-channel mask 删除仍保留。

### DEC-034 补充（2026-08-14）

- **状态**：`Accepted Design / Not Implemented`（文档已更新；当前代码仍需按该
  结构重构）。
- **选择**：
  1. 正式 V1 中不再使用 `A_query` 作为主线 action token。Action/policy 主变量
     统一命名为 `A_noise` / `A^σ`，训练时来自 GT action chunk 加噪，推理时来自
     action prior/noise。
  2. `H_action = H_nav = 10`。视频数据侧 pseudo motion/action 重采样到 10 个
     slots；VLN 侧预测 10 个 low-level actions。闭环推理默认执行第一个 action
     后更新 observation/Register。
  3. Action output 采用 GigaWorld / DreamZero 常见做法：
     `A_noise + context -> action stream/action expert -> action decoder ->
     predicted action velocity/noise -> scheduler/flow denoise -> A_out`。
     离散 CE / STOP balance 可作为 auxiliary 或最终 discrete decode 对齐，但不再
     是唯一主结构。
  4. Backbone 正式改为 dual-stream / MoT-style：visual stream 继承 Wan2.1 video
     DiT token 计算，action stream 使用独立 action expert / decoder，并通过受控
     attention 或 cross-stream adapter 交互。single hidden-width 结构只作为历史
     smoke/ablation。
  5. Causal attention 关系冻结：`A_noise` 可读 Register、obs、instruction、
     `A_hist` 和自身；不可读 `Z_future^σ`、future 3D GT、clean `A_cur` 或 clean
     target action。`Z_future^σ` 可读 Register、obs、instruction、`A_noise`
     hidden 和 `A_cur` generation condition。`A_cur` 只允许服务 video generation
     future tokens，不得反向污染 Register/obs/action tokens。
- **理由**：GigaWorld 和 DreamZero 的 action path 都说明，将动作当作 noisy
  variable 并通过专门 action decoder 做 flow/diffusion decode，是 video generation
  与 policy/action cotrain 更通用、更干净的形式。它避免了 `A_query` learnable
  token 语义含混，也避免 direct CE 旁路 head 破坏 shared backbone 设定。
- **影响文档**：`00_overview/project_invariants.md`、
  `01_design/v1_action_centered_io_interface.md`、
  `01_design/v1_open_design_questions.md`、
  `04_training/v1_three_stage_training_data_plan.md`。
- **替代关系**：替代 DEC-031/DEC-033 中 `A_query/A_noise` 混用、single hidden-width
  smoke 作为近似正式结构、以及 VLN direct CE logits 作为主输出的口径；DEC-032
  的 16-channel patch stem、删除 InfiniteWorld `20-channel mask`、token/type
  identity 显式表达继续保留。

### DEC-035 补充（2026-08-14）

- **状态**：`Accepted Design / Not Implemented`（文档已更新；代码中若仍有
  `A_hist` 调制 Extractor / Updater，需要在下一次结构实现时删除）。
- **选择**：Register memory 改为 visual-only。具体地：
  `R_0 = Extractor(C_0)`，`R_{i+1}=Updater(R_i, C_i)`；Extractor / Updater 不接受
  `A_hist`、pose、odometry、ego-motion 或 pseudo motion。历史 action/motion
  可以继续落盘，用于审计、尺度校准、Stage Three action supervision 或 Register
  外部的独立 action/state context，但不得被融合进 Register 更新。
- **理由**：Register 的核心角色应是视觉历史状态，而不是动作/pose 先验容器。
  把 `A_hist` 写进 Register 会让 memory 语义变混，并增加后续真实世界无 pose /
  无可靠 action 数据使用时的训推不一致风险。动作因果关系留给 `A_noise` action
  stream、`A_cur` generation condition 和 Stage Three action loss 处理。
- **影响文档**：`00_overview/project_invariants.md`、
  `01_design/v1_action_centered_io_interface.md`、
  `01_design/v1_open_design_questions.md`、
  `04_training/v1_three_stage_training_data_plan.md`、
  `04_training/v1_stage_one_t4_iw_aligned.md`。
- **替代关系**：覆盖 DEC-027/DEC-031 以及 DEC-034 中所有可能暗示 `A_hist`
  参与 Register update 的表述；不改变 DEC-034 的 `A_noise`、`H_action=10`、
  dual-stream backbone 和 action flow decoder 选择。

### DEC-036 补充（2026-08-14）

- **状态**：`Accepted Design / Not Implemented`（文档已更新；代码下一步需按该
  口径实现）。
- **选择**：覆盖 DEC-035。`A_hist` 重新参与 Register Extractor / Updater：
  `R_0 = Extractor(C_0, A_hist_0)`，
  `R_{i+1}=Updater(R_i, C_i, A_hist_i)`。但 `A_hist` 必须以独立 action tokens、
  cross-attention context 或 gated token adapter 的方式与视觉 tokens 交互。
  明确禁止使用 action bias / latent bias / additive bias：不得把 action embedding
  直接加到 video latent、patch tokens、Register tokens 或 DiT hidden 上。
- **理由**：历史动作确实有助于解释历史视觉变化，特别是同一视觉变化可能来自
  不同自运动/控制。但 bias 方案会把动作语义粗暴写进视觉 latent/Register，容易
  造成 memory 表示混浊，也不利于后续和 `A_noise` action stream、dual-stream
  backbone 做清晰因果分工。token/cross-attention 方案更干净：动作可被读，但不是
  视觉状态本身。
- **影响文档**：`00_overview/project_invariants.md`、
  `01_design/v1_action_centered_io_interface.md`、
  `01_design/v1_open_design_questions.md`、
  `04_training/v1_three_stage_training_data_plan.md`、
  `04_training/v1_stage_one_t4_iw_aligned.md`、
  `00_overview/project_status.md`。
- **替代关系**：直接覆盖 DEC-035；保留 DEC-034 的 `A_noise`、`H_action=10`、
  action flow decoder、dual-stream / MoT-style backbone 和 causal leakage 规则。

### DEC-037 补充（2026-08-14）

- **状态**：`Accepted / Implemented in full-pipeline smoke`（
  `src/nav/v1/models/full_model.py` 已实现统一 `RegisterCell`）。
- **选择**：Register 侧不再区分 `Extractor` 与 `Updater`。统一为一个
  recurrent `RegisterCell`：

  ```text
  R_{-1} = R_null
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  ```

  其中 `R_null` 是 fixed、non-episode-specific 的 register template，只提供
  register slot/type/position 结构，不携带可学习场景先验。`C_0` 的首步写入和后续
  `C_i` 的递归更新都经过同一个 `RegisterCell`。`A_hist` 继续以独立 action tokens /
  cross-attention context / gated token adapter 参与；继续禁止 action bias /
  latent bias / additive bias。
- **理由**：统一 cell 可以消除“首步 Extractor 与后续 Updater 是否学到不同语义”的
  结构分叉，也自然解决 `R_0` 来源问题：`R_0` 不是 learnable initial memory，而是
  `R_null` 经过第一个 observation/action 写入后的在线 memory。
- **影响文档**：`00_overview/project_invariants.md`、
  `01_design/v1_action_centered_io_interface.md`、
  `01_design/v1_open_design_questions.md`、
  `04_training/v1_three_stage_training_data_plan.md`、
  `04_training/v1_stage_one_t4_iw_aligned.md`、
  `00_overview/project_status.md`。
- **替代关系**：覆盖 DEC-036 中 `Extractor / Updater` 的双模块表述；保留 DEC-036
  的 `A_hist` token/cross-attention 参与和 no-bias 约束，也保留 DEC-034 的
  `A_noise`、`H_action=10`、action flow decoder、dual-stream backbone。

### DEC-038 补充（2026-08-14）

- **状态**：`Accepted / Implemented / Smoke Verified`。
- **选择**：当前 NAV 主线可执行代码只保留新版完整模型链路。旧 scaffold、小
  bypass head、旧 A/B Register、旧 `A_query`、旧 action-bias 和旧
  InfiniteWorld adapter 入口从当前代码删除；历史结果保留在 v0 文档和 git 历史。
- **完整模型 smoke**：

  ```text
  script:
    NAV/scripts/run_v1_full_pipeline_smoke.sh

  model:
    NAV/src/nav/v1/models/full_model.py

  latest report:
    NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json
  ```

- **验证范围**：
  1. Stage One：`L_visual_flow` forward/backward/update；
  2. Stage Two：`L_stage1 + λ_3d L_3D` forward/backward/update；
  3. Stage Three：`L_action_flow + λ_ce CE + λ_video L_visual_replay +
     λ_3d L_3D_replay` forward/backward/update；
  4. videogen inference：输出 `z_future` / `future_velocity`；
  5. policy inference：不输入 future noisy video，输出 `action_chunk` /
     `primitive_logits` / `primitive_ids`；
  6. 结构审计：必须使用 `RegisterCell`，不得出现 Extractor/Updater 模块，不得
     出现 action/latent additive bias path。
- **理由**：之后所有训练、测试、推理冒烟测试都必须使用完整代码和完整数据流；
  只跑 scaffold 或旁路小结构会再次把速度、loss 和可行性结论带偏。
- **影响文档**：`doc/AGENT.MD`、`05_evaluation/v1_full_pipeline_smoke.md`、
  `00_overview/project_status.md`、`06_operations/resource_inventory.md`。
- **替代关系**：覆盖 DEC-031/032/033 的旧 `A_query`/Wan patch 实现入口；保留
  其中被 DEC-034/037 吸收的原则，例如 no 20-channel mask、per-token timestep、
  policy-safe attention 和官方 Wan 初始化方向。


新增决策时写明日期、状态、选择、主要理由、影响文档和替代关系。只有已经确认并
落实到设计或配置的事项标记为 `Accepted`；讨论中的选项使用 `Proposed`。
