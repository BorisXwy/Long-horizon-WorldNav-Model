# NAV 关键决策日志

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-002` |
| 类型 | 决策日志（Decision Log） |
| 状态 | Live |
| 更新时间 | 2026-09-05 |
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
| DEC-056 | 2026-09-03 | Accepted / Running | GigaNav Wan2.1-1.3B ablation 的导航 action horizon 固定为 H=8；正式训练采用物理 BS=1、梯度累积 32、EBS=32，保持完整共享 Wan backbone 与四类 CE。训练通过 Infinite-World 虚拟环境优先启用 FlashAttention；不可用时仅退回 SDPA，不改变结构。GPU1 启动独立长训，GPU0 不受影响。 |
| DEC-057 | 2026-09-05 | Superseded / Corrected in DEC-058 | 曾误把任务理解为修改标签并重训 reference-frame policy；该 alignment-v2 run 已在 step13 停止且无正式 checkpoint，不再使用。 |
| DEC-058 | 2026-09-05 | Accepted / Code Updated / Evaluated | 保留已有 GigaNav `step_005000.pt` 与 `+12` action target 训练语义，改闭环推理为滚动 13 帧 observation window：时刻 `t` 的 T4 首 causal plane 对应 `O(t-12)`，输出首动作执行为 `A(t)`；episode 起始 12 步用首帧左填充并从分段统计中排除。R2R-train 前20条评测 SR=0、Oracle Success=5%，early/middle/late 平均 goal progress 均为负，模型 94.91% 输出 MOVE_FORWARD 且从不 STOP，未形成聚合意义的前中期方向跟随。 |
| DEC-017 | 2026-07-30 | Running | Stage One 1.0改为GPU0训练A、GPU1训练B；GPU0保留1个latent worker。启动时GPU1被其他账号占用约41GiB，B进入45,000MiB free-memory等待队列，启动并完成首步后自动增加3个GPU1 latent workers |
| DEC-018 | 2026-07-31 | Accepted | Stage One 1.0正式训练从1000扩展为10000 optimizer steps，每1000 steps保存完整checkpoint；旧A-1000在step373停止并作为历史run保留，新A/B-10000均从原始Infinite-World重新初始化 |
| DEC-019 | 2026-08-03 | Accepted | Stage One 1.0正式训练改为1000 effective optimizer steps、effective batch size 16（A: micro=1×accum=16；B: micro=2×accum=8），每100 effective steps保存完整checkpoint；A/B均from scratch从原始Infinite-World初始化，不加载旧RE10K或旧A/B-10000 checkpoint；其余显存继续并行准备full_episodes_v1 latent |
| DEC-020 | 2026-08-07 | Accepted / In Progress | 闭环完整训练主数据固定为 R2R-CE、RxR-CE、LHPR-VLN、ScaleVLN 四套，统一落盘 `/sharedata/datasets/{R2R,RxR,LHPR-VLN,ScaleVLN}`；以 CE/任务标注为训练入口，原始 RxR GCS pose traces 与 StreamVLN CE 子集为可选增强 |
| DEC-021 | 2026-08-07 | Accepted / In Progress | R2R-CE/RxR-CE 外部 SOTA 复现只纳入文献 R2R Val-Unseen SR≥StreamVLN（含 StreamVLN）且有公开权重的工作；仓库 clone 到 `3d_wm_vln/`，环境用 `virtual_env/.venv_*` venv；协议见 NAV-EVL-002 |
| DEC-022 | 2026-08-08 | Proposed / Working Hypothesis | Stage Three：生成+Register 改 NavPolicy 时，训练侧按接近视频 chunk 的**空间变化尺度**更新 Register（观测常仅 ~4 frame）；推理快则可单步决策，慢则与 Register 更新同步成半闭环。控制 chunk≠81 帧 VAE chunk。详见 NAV-DES-003、insight R17 |
| DEC-023 | 2026-08-09 | Accepted Design / Not Implemented | NAV 采用 action-centered causal interface：future visual / future 3D 只作为动作和Instruction/Goal条件下的 consequence supervision，不作为 policy condition；Stage One/Two/Three 共享同一 DiT/Register backbone，Stage Three 使用其 policy-only裁剪路径而非新建 perception backbone |
| DEC-024 | 2026-08-09 | Accepted Design / Not Implemented | 新版 NAV 中 history/Register 默认作为 visual/main stream 的 video tokens 进入 DiT，而不是作为 text/condition 侧静态 token；B-style condition Register 只保留为 ablation。详见 NAV-DES-004 |
| DEC-025 | 2026-08-10 | Accepted Design / Not Implemented | V1 中 current/target action 不作为 policy condition；action/nav tokens 是 policy 要生成的 query/noised variables。Generation branch 可读取 shared WanBlock 内 action-token hidden 来生成 future consequence；history 侧只可使用 previous executed action。详见 NAV-DES-004/005 与 NAV-TRN-006 |
| DEC-026 | 2026-08-10 | Accepted / Smoke Verified | V1 Stage One generation horizon 固定为 `T_latent=4`：每个 micro chunk 使用13帧、stride 12独立 VAE encode；与 InfiniteWorld 的对齐只对齐 history/context span，IW-1/4/8/16 chunks 分别换算为7/27/54/107个 micro history steps；target loss 仍只算 T4 short future。详见 NAV-TRN-007 |
| DEC-034 | 2026-08-14 | Partially Superseded by DEC-041 | V1 action path 从 `A_query` 正式改为 `A_noise`；`H_action=H_nav=10`；动作输出仿照 GigaWorld/DreamZero 的 action flow/diffusion decode；其中“backbone 改为 dual-stream / MoT-style”已被 DEC-041 废弃。 |
| DEC-035 | 2026-08-14 | Accepted Design / Not Implemented | Register Extractor / Updater 不再读取 `A_hist`、pose、odometry 或 pseudo motion；Register 只由历史/current visual observation 更新。`A_hist` 若保留，只能作为 Register 外部 action/state context、训练监督或审计字段。 |
| DEC-036 | 2026-08-14 | Accepted Design / Not Implemented | 覆盖 DEC-035：`A_hist` 重新参与 Register Extractor / Updater，但只能以独立 action tokens / cross-attention context 交互；禁止 action bias / latent bias / additive bias 方案。 |
| DEC-037 | 2026-08-14 | Accepted Design / Not Implemented | 覆盖 DEC-036 的双模块表述：Register 侧统一为 `RegisterCell`。每个 episode 从 fixed `R_null` 开始，首步和后续都执行 `R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`；继续禁止 action/latent/additive bias。 |
| DEC-041 | 2026-08-14 | Accepted / Code Updated / Training Pending | V1 主结构固定为 single shared WanBlock token stream，不采用 GigaWorld-style dual-stream / MoT-style backbone；Register、obs、future、text、A_hist/A_cur/A_noise 通过同一 Wan2.1-style DiT blocks、token type、timestep 与 causal mask 组织。保留最新 `A_noise`、`H_action=10`、action flow decode、Stage2 current hidden pose supervision 等实现结论。 |
| DEC-042 | 2026-08-15 | Accepted / Docs Consolidated | NAV 文档从大量 V0/V1 碎片文件合并为主题总文档：规则、模型、数据、训练、评测、insight、资源。旧独立文件并入章节，不再作为事实源维护。 |
| DEC-043 | 2026-08-17 | Accepted / Code Updated / Full-model 1-step Verified | V1 video branch 恢复 InfiniteWorld/Wan RFlow 口径：`x_t=(1-t)z_target+t noise`，`future_velocity` 监督 `noise-z_target`（`use_reversed_velocity=true`），推理采样用 `z <- z - v_rev * dt`。该改动只修正 video branch 的 noisy latent / velocity target / sampler；RegisterCell、A_hist/A_cur/A_noise、shared token interaction、pose/action heads 不改变。 |
| DEC-044 | 2026-08-22 | Accepted / Operational Rule Updated | 为避免公共 `/sharedata` 被训练 latent 挤满，原始数据和 simulator assets 继续放在 `/sharedata`，但新的训练用 latent / tensor cache 真实落盘位置改为 `NAV/data/train/<dataset>/<latent_run>/`。`/sharedata/NAV/derived/` 只保留历史遗留、预算 manifest、轻量索引和必要临时 render 中间态。 |
| DEC-045 | 2026-08-25 | Accepted / Code Updated / Training Pending | Stage3 R2R policy 通过在 dataloader 内复制包含较多 TURN/STOP target 的真实 window 改善动作分布，loss 保持普通 CE；正式 accuracy eval 仍默认使用不复制的 natural R2R window 分布。新 Stage3 必须从 Stage2 step2600 初始化，替代已出现 MOVE collapse 的 step1600-init run。 |
| DEC-046 | 2026-08-25 | Accepted / Code Updated / Training Pending | Stage3 只替换 backbone 之后的离散 action readout：加载完整 Stage2 step2600 后，从 Wan 最后一层现有 action-output slots 读取 `[B,10,1536]` hidden，经 fresh `Linear(1536,144)` 输出 combo logits。backbone 之前及内部的输入、A_noise/timestep、token layout 和 mask 全部不变；旧 flow output heads 冻结且不进入 loss。 |
| DEC-047 | 2026-08-27 | Superseded by DEC-052 / Stopped at step 148 | Stage3 R2R 的每个 policy target 必须读取从 episode 起点到当前 `Z_obs` 之前的完整前缀：所有样本满足 `start_micro=0`、`history_micro=obs_micro`，完整前缀只通过固定大小 Register recurrent update 压缩，不向 Wan 拼接原始历史 token。原实验保持四类均衡 one-step CE；`micro=2 × accum=8` OOM 后改用 `micro=1 × accum=16`。完整历史语义继续保留，但 one-step 均衡采样口径由 DEC-052 覆盖。 |
| DEC-052 | 2026-08-28 | Accepted / Code Updated / Running | 正式 Stage3 R2R 改为 `action_chunk=4`，但 backbone 永远保留 Stage2 的 10 个 action token。`H` 只控制读取和监督前 H 个 hidden：H=4 时前 4 个 hidden 分别经共享四分类 MLP，loss 为四个位置的平均 CE；后 6 个 token 仍参与原 backbone 前向但不读出、不监督。训练 window 按自然分布 shuffle、无放回遍历，不做类别均衡、复制或 class weight。完整 episode prefix/Register 语义、Stage2 visual/pose replay、EBS16 和 step3400 初始化均不变。原 full-history balanced one-step run 在 step148 停止；此前短窗口 balanced one-step 对照继续运行。 |
| DEC-053 | 2026-08-28 | Measured / H4 Training Continues | 旧 H1 balanced short-window 训练在 step1418 停止，最新完整 checkpoint 为 step1400。使用 256 个同序、无均衡的 natural full-prefix R2R windows 成对评测旧 H1 step1400 与新 H4 step200：旧 H1 slot-0 accuracy=21.48%、macro recall=34.02%；新 H4 slot-0 accuracy=73.44%，但等于 MOVE majority baseline，且四槽 1024/1024 个预测均为 MOVE_FORWARD，macro recall=25%。因此 H4 当前较高 accuracy/较低 CE 是多数类塌缩而非有效 action learning；H4 正式训练暂继续，后续 checkpoint 必须用同一协议复测。 |
| DEC-054 | 2026-09-01 | Accepted / Code Updated / Running | 停止 DEC-052 的无权重 natural H4 训练；新对照仍按 35,941 个 full-prefix unique windows 自然 shuffle、无放回遍历且不复制样本，但 policy objective 改为 inverse-frequency class-balanced CE。权重由全训练集 H4 token 计数计算为 `w_c=N/(4N_c)`，逐 token 加权后直接求 mean；Stage2 step3400 初始化、10 个 backbone action slots/前4个监督、EBS16、visual/pose replay 与优化器参数均不变。完整 step1 已通过，未出现启动即全 MOVE。 |
| DEC-055 | 2026-09-02 | Accepted / Ablation Implemented / Full-chain Smoke Verified | 新增独立 `GigaNav` 导航消融：核心代码统一放 `src/giga_nav/`，脚本放 `scripts/`；使用本地官方 Wan2.1-T2V-1.3B（约1.421B backbone params），保留 GigaWorld 的 `[state, reference visual, action, noisy future]` 共享 token 顺序和 action-only readout。因 R2R 标签是离散导航动作，唯一任务侧改动为四类 CE，缺失 proprioception 时显式零 state token；不得将其当作 V1 Register/3D 正式主线。 |

## 记录要求

### DEC-054 补充（2026-09-01）

- **停止旧任务**：无权重 H4 continuation 在内存 step2375 左右停止；最后一个
  完整 checkpoint 为 `step_002200.pt`。step1–2375 的 accuracy 长期约等于
  MOVE_FORWARD 占比，macro recall 约 25%，因此不能继续把 loss 下降解释为
  conditional policy learning。【已验证→log/v1_stage3_r2r_single_action/stage3_h4_natural_resume2000_to6000_gpu0_20260831_114329/train.jsonl】
- **不改数据采样**：仍使用 35,941 个 natural full-prefix unique windows，epoch
  内 shuffle 后无放回遍历；不复制 TURN/STOP window，不做四类 cyclic sampler。
- **权重口径**：全量 `H=4` token 计数为
  `STOP/MOVE/LEFT/RIGHT=9274/96177/18900/19413`，总数 `N=143764`；loss 使用
  `w_c=N/(4N_c)`，对应约 `3.875/0.374/1.902/1.851`。实现必须计算
  `mean(CE_token * w_target)`，不得使用 physical batch=1 下会按当前 microbatch
  权重和重新归一化的 `F.cross_entropy(weight=..., reduction="mean")`。
- **严格对照**：新 run 从与 DEC-052 相同的 Stage2 step3400 初始化，不继承
  majority-collapse 的 Stage3 权重或 optimizer；除 policy loss 外，模型结构、
  full-history Register、instruction、`Z_obs`、H4 readout、Stage2 visual/pose replay、
  `micro=1 × accumulation=16` 与 learning rate 全部保持一致。
- **验收重点**：raw accuracy 不作为首要指标；必须检查 macro recall、四类 recall、
  prediction distribution，以及 correct/shuffled/empty instruction sensitivity。
- **启动核验**：新 run 的完整 step1 用时 145.80 秒，CUDA max allocated
  28.23 GiB；weighted policy loss=1.2691、unweighted CE=1.4363、visual
  replay=0.1043、pose replay=1.15e-4。64 个动作预测分布为
  `STOP/MOVE/LEFT/RIGHT=11/12/40/1`、macro recall=32.22%，已脱离旧任务从启动
  即全 MOVE 的模式，但单步不构成收敛或泛化结论。
  【已验证→log/v1_stage3_r2r_single_action/stage3_r2r_fullhistory_natural_h4_classbalanced_from_stage2step3400_6k_20260901/train.jsonl】


### DEC-052 补充（2026-08-28）

- **动作目标**：`H_action=4`，target shape 为 `[B,4]`；第 `h` 个 slot
  监督 `label_start_action+h` 的真实离散动作。若 chunk 内遇到 episode 的首个
  `STOP`，该位置及其后续 padding 均为 `STOP`。
- **输入/输出**：输入侧始终保留 Stage2 的 10 个确定性 zero-noise、
  zero-timestep action query slots，backbone 输出 `[B,10,1536]`。当 `H=4` 时只取
  `hidden[:,0:4,:]`，由同一个 FP32 MLP 逐位置输出 `[B,4,4]` logits；后 6 个
  hidden 不读出、不计算 loss。禁止增加 `10→4` 跨槽位 temporal projector，
  `H` 的变化不得改变 backbone token layout。Stage2 checkpoint 的
  Wan/Register/video/pose/action-token 输入权重原样继承，policy MLP 随机初始化。
- **采样**：一个 episode 中每个有效 `Z_obs` 只构造一个 full-prefix window；所有
  unique windows 每个 epoch shuffle 后无放回遍历。不再按 STOP/MOVE/LEFT/RIGHT
  做 cyclic balance，不复制 TURN/STOP 样本，也不使用 class weight。
- **运行状态**：被替代的
  `stage3_r2r_fullhistory_mb1_ebs16_from_stage2step3400_2k_20260827`
  已在 step148 停止；旧短窗口对照
  `stage3_r2r_single_action_balanced_from_stage2step3400_2k_20260826`
  保持运行。新 H4 正式任务已于 2026-08-28 02:09 在 GPU0 启动，入口为
  `scripts/run_v1_stage3_r2r_full_history_ebs16.sh`，TensorBoard 端口为 `6040`。
  首个 optimizer step 用时 140.14 秒，policy CE=1.4363、visual replay=0.1043、
  pose replay=1.15e-4，CUDA max allocated=28.23 GiB；optimizer state 建立后
  GPU0 进程总占用约 31.4 GiB，未发生 OOM。

### DEC-053 补充（2026-08-28）

- **统一评测入口**：`scripts/eval_v1_stage3_action_chunk_compare.py`。
- **checkpoint**：旧 H1 使用
  `stage3_r2r_single_action_balanced_from_stage2step3400_2k_20260826/checkpoints/step_001400.pt`；
  新 H4 使用
  `stage3_r2r_fullhistory_natural_a4_mb1_ebs16_from_stage2step3400_2k_20260828/checkpoints/step_000200.pt`。
- **协议**：R2R train、256 个 paired unique windows、full episode prefix、natural
  shuffle、无 class balancing、correct instruction、batch size 4、完整
  Register + shared Wan + policy head 前向。旧 H1 只评 future slot 0；新 H4
  评 4 slots，并单列 slot 0 作一一配对比较。
- **旧 H1**：slot-0 CE=1.3362、accuracy=21.48%、macro recall=34.02%；预测
  STOP/MOVE/LEFT/RIGHT=`70/38/96/52`，没有单类塌缩，但远低于 natural
  MOVE majority baseline 73.44%。该结论只适用于 full-history natural 分布，
  不等价于其原生 balanced short-window 训练分布表现。
- **新 H4**：slot-0 accuracy=73.44%、macro recall=25%；全四槽共 1024 actions，
  CE=0.9603、accuracy=66.21%、sequence exact match=20.70%。所有 1024 个预测
  均为 MOVE_FORWARD，因此 slot-0 和全四槽 accuracy 都恰好等于各自 majority
  baseline；STOP/LEFT/RIGHT recall 全为 0。
- **结论**：step200 的 H4 仍处于 natural-distribution majority collapse；不能将
  较低 CE 或较高 raw accuracy 解释为导航能力提升。H4 训练继续，后续 step400+
  checkpoint 必须复用同一 paired protocol 检查 macro recall 与 rare-action recall。
- **结果**：`result/v1_stage3_action_chunk_compare/paired256_old1400_vs_h4step200_20260828/summary.json`。
  【已验证→result/v1_stage3_action_chunk_compare/paired256_old1400_vs_h4step200_20260828/summary.json】
- **旧 H1 纵向核验**：使用完全相同的原生 short-window natural evaluator、
  seed=20260830 和 128 windows 对比 step800/step1400。step800 为
  CE=1.3601、accuracy=15.63%、macro recall=26.80%，step1400 为
  CE=1.4696、accuracy=14.84%、macro recall=26.55%；预测分布分别为
  `13/7/108/0` 与 `11/7/110/0`（STOP/MOVE/LEFT/RIGHT）。前 32 个落盘 example
  的 sample ID/history 完全配对。因此 balanced training loss 的下降没有转化为
  natural policy 改善，step800 后继续训练到 step1400 基本无效。
  【已验证→result/v1_stage3_single_action/step1400_natural128_seed20260830/summary.json】

### DEC-046 补充（2026-08-25）

- **权重加载**：先按原 Stage2 图严格加载 step2600，再挂接 Stage3 专用
  `Linear(hidden_dim=1536, combo_dim=144)`；因此 shared Wan backbone、RegisterCell、
  video branch、pose head 与 action-token input path 均继承 step2600，只有新离散
  policy head 随机初始化。
- **数据流**：backbone 输入侧完全不改，未来 10 个 action slot 仍按既有
  `A_noise + action timestep + position` 编码和既有 attention mask 经过 shared Wan。
  唯一改动位于 backbone 之后：直接切出 Wan 最后一层现有 action-output 位置的
  10 个 hidden token `[B,10,1536]`，分别经同一个 Linear 得到 `[B,10,144]`
  logits。
- **loss**：Stage3 action loss 只有普通 combo CE；`loss_action_flow=0` 只作为兼容
  日志字段。旧 action flow/classification output modules 冻结且不参与 forward decode，
  visual/pose replay 保持不变。
- **兼容性**：Stage2 续训仍使用原 `iw_flow_combo` 配置，不受该 Stage3-only head
  改造影响；新的 Stage3 checkpoint 在 `model_config.policy_head_type=linear` 中记录
  构图方式，可由 evaluator 原样恢复。

### DEC-045 补充（2026-08-25）

- **动机**：旧 Stage3 step600 在 256-window natural-distribution 配对评测中，
  `MOVE_FORWARD` 占预测 `2499/2560`，`TURN_LEFT/RIGHT recall=0`；正确、打乱、
  空 instruction 的 top-1 预测几乎不变。证据位于
  `result/v1_stage3_open_loop_policy/r2r_train_step600_instruction_ablation_256_20260825/summary.json`。
- **window sampling**：只复制 dataloader 内的 window 引用，不复制 latent 文件。
  对一个未来 10-step target，复制数为
  `min(10, 1 + 2*n_turn + 2*n_stop)`；普通 MOVE-only window 保持一份，含更多
  TURN/STOP 的真实 window 获得更多副本。复制上限避免少量样本无限主导训练。
  在当前 69,648 个 unique window 上形成 478,951 个虚拟 window；按训练时
  history 长度均匀采样的精确预期 token 比例由
  `STOP/MOVE/LEFT/RIGHT=15.91/59.58/12.42/12.09%` 改善为
  `20.44/51.82/13.97/13.77%`。
- **loss**：保持普通、无 class weight 的 combo CE。动作比例的改善完全来自
  dataloader 样本构造，便于直接审计训练实际看到的数据分布。
- **评测口径**：训练可以 action-balanced；报告 accuracy、macro recall、
  instruction ablation 时默认恢复 natural window sampling，避免在人工均衡分布上
  抬高指标。
- **替代关系**：替代
  `stage3_r2r_future_action_from_stage2step1600_2k_20260823`；旧 run 与 checkpoint
  只保留为失败诊断，不作为后续 Stage3 初始化。

### DEC-044 补充（2026-08-22）

- **状态**：`Accepted / Operational Rule Updated`。
- **选择**：项目内新增 `NAV/data/train/<dataset>/<latent_run>/` 作为训练 latent
  的真实落盘根目录；这不是 symlink 入口，也不是 Git-tracked 数据目录。
  `/sharedata` 保持为原始数据、公共资产、轻量 manifest 与临时 render 缓存所在位置。
- **理由**：Stage One/Two/Three 的 T4 latent、text embedding 和 VLN policy
  latent 会达到数百 GiB 量级，继续写入 `/sharedata/NAV/derived/` 会挤占公共
  数据盘并干扰原始数据下载/渲染。
- **已执行**：2026-08-22 已将当前 VLN stoppad stream encoder 的输出从
  `/sharedata/NAV/derived/v1/vln/t4_micro_latents_500g_stoppad/...` 切到
  `NAV/data/train/rxr_ce/t4_micro_latents_stoppad_20260822_1423/`，已生成内容
  同步移动，encoder 续跑。
- **影响文档**：`AGENT.md`、`README.md`、
  `doc/00_overview/project_rules_and_documentation_standard.md`、
  `doc/02_data/data_preparation_schema_and_status.md`、
  `doc/06_operations/resource_inventory.md`。

### DEC-042 补充（2026-08-15）

- **状态**：`Accepted / Docs Consolidated`。
- **选择**：将旧 `01_design/`、`02_architecture/`、`03_data/`、`04_training/`、
  `05_evaluation/`、`07_research/`、`08_insight/` 的碎片文档按主题合并为：
  `01_model/model_evolution_and_current_architecture.md`、
  `02_data/data_preparation_schema_and_status.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `04_evaluation/evaluation_reproduction_and_benchmarks.md`、
  `05_insight/insight_log.md`；规则类内容合并到
  `00_overview/project_rules_and_documentation_standard.md`。
- **理由**：原文档数量过多，跨文件引用松散，模型结构、接口、训练计划和数据准备
  之间的事实源容易分裂。主题总文档便于按“模型—数据—训练—评测—资源”维护。
- **影响文档**：`README.md`、`AGENT.md`、`00_overview/project_status.md`、
  `06_operations/resource_inventory.md` 以及上述主题总文档。
- **替代关系**：旧独立文件内容已并入对应章节；后续新增内容优先更新主题总文档。

### DEC-022 补充（2026-08-08）

- **状态**：`Proposed`（工作设想登记，未实现、未冻结 packing/损失）。
- **选择**：Nav 适配时 Register 更新对齐「chunk 级空间位移」而非强凑 81 RGB；推理频率可按算力在单步与 Register 同步之间切换。
- **理由**：视频帧间微动 vs Habitat 大步进的数量级对照表明，~4 Habitat step 的位移接近 1 个室内 video chunk，适合半闭环局部原语；与 DEC-014（真实 Observation History）不冲突。
- **影响文档**：`01_model/model_evolution_and_current_architecture.md`（NAV-DES-003）、`05_insight/insight_log.md`（R17）。
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
- **影响文档**：`01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`。
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
- **影响文档**：`01_model/model_evolution_and_current_architecture.md`。
- **替代关系**：细化并部分替代 DEC-002 中 B 方案的默认地位；B-style
  condition Register 仍可作为 ablation，不作为新版主方案。

### DEC-025 补充（2026-08-10）

- **状态**：`Accepted Design / Not Implemented`（语义已写入
  `NAV-DES-004/005` 与 `NAV-TRN-006`，代码接口和 mask 尚未实现）。
- **选择**：当前/目标 action 不作为 policy condition。Policy 侧的
  action/nav tokens 是 query 或 noised target，是模型要生成/去噪/分类的变量。
  Generation branch 可以读取 shared WanBlock 内的 action-token hidden state，
  用于预测该 action 导致的 future visual consequence。
- **理由**：若 clean target action 同时作为输入和输出，会造成 policy 侧语义
  泄漏；但 generation branch 又必须知道 action，才能学习 action-token-conditioned
  future consequence。把 action 放在 shared WanBlock 内部 action tokens，既避免 policy
  leakage，又保留 video generation 所需的 action-stream 条件。
- **影响文档**：`01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`。
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
- **影响文档**：`03_training/training_plan_and_experiment_log.md`、
  `02_data/data_preparation_schema_and_status.md`、
  `03_training/training_plan_and_experiment_log.md`。
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
- **影响文档**：`01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`。
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
- **影响文档**：`03_training/training_plan_and_experiment_log.md`。
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
- **影响文档**：`03_training/training_plan_and_experiment_log.md`。
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

- **状态**：`Superseded by DEC-034/037/038/041`（旧 `A_query` 实现已删除；
  主线改为 `A_noise` + single shared WanBlock + `RegisterCell`）。
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
  结构重构）。其中第 4 点 dual-stream backbone 已被 DEC-041 覆盖。
- **选择**：
  1. 正式 V1 中不再使用 `A_query` 作为主线 action token。Action/policy 主变量
     统一命名为 `A_noise` / `A^σ`，训练时来自 GT action chunk 加噪，推理时来自
     action prior/noise。
  2. `H_action = H_nav = 10`。视频数据侧 pseudo motion/action 重采样到 10 个
     slots；VLN 侧预测 10 个 low-level actions。闭环推理默认执行第一个 action
     后更新 observation/Register。
  3. Action output 采用 GigaWorld / DreamZero 常见做法：
     `A_noise + context -> shared WanBlock action-token hidden -> action decoder ->
     predicted action velocity/noise -> scheduler/flow denoise -> A_out`。
     离散 CE / STOP balance 可作为 auxiliary 或最终 discrete decode 对齐，但不再
     是唯一主结构。
  4. 【已废弃，见 DEC-041】Backbone 改为 dual-stream / MoT-style 的选择不再作为
     NAV 正式主线。正式主线回到 single shared WanBlock token stream。
  5. Causal attention 关系冻结：`A_noise` 可读 Register、obs、instruction、
     `A_hist` 和自身；不可读 `Z_future^σ`、future 3D GT、clean `A_cur` 或 clean
     target action。`Z_future^σ` 可读 Register、obs、instruction、`A_noise`
     hidden 和 `A_cur` generation condition。`A_cur` 只允许服务 video generation
     future tokens，不得反向污染 Register/obs/action tokens。
- **理由**：GigaWorld 和 DreamZero 的 action path 都说明，将动作当作 noisy
  variable 并通过专门 action decoder 做 flow/diffusion decode，是 video generation
  与 policy/action cotrain 更通用、更干净的形式。它避免了 `A_query` learnable
  token 语义含混，也避免 direct CE 旁路 head 破坏 shared backbone 设定。
- **影响文档**：`00_overview/project_rules_and_documentation_standard.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`。
- **替代关系**：替代 DEC-031/DEC-033 中 `A_query/A_noise` 混用以及 VLN direct
  CE logits 作为主输出的口径；dual-stream backbone 子决策由 DEC-041 废弃。
  DEC-032 的 16-channel patch stem、删除 InfiniteWorld `20-channel mask`、
  token/type identity 显式表达继续保留。

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
- **影响文档**：`00_overview/project_rules_and_documentation_standard.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `03_training/training_plan_and_experiment_log.md`。
- **替代关系**：覆盖 DEC-027/DEC-031 以及 DEC-034 中所有可能暗示 `A_hist`
  参与 Register update 的表述；该条随后又被 DEC-036/037 覆盖。Backbone 口径
  最终遵循 DEC-041。

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
  造成 memory 表示混浊，也不利于后续和 `A_noise` action tokens、shared WanBlock
  backbone 做清晰因果分工。token/cross-attention 方案更干净：动作可被读，但不是
  视觉状态本身。
- **影响文档**：`00_overview/project_rules_and_documentation_standard.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `00_overview/project_status.md`。
- **替代关系**：直接覆盖 DEC-035；保留 DEC-034 的 `A_noise`、`H_action=10`、
  action flow decoder 和 causal leakage 规则；backbone 口径遵循 DEC-041。

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
- **影响文档**：`00_overview/project_rules_and_documentation_standard.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `01_model/model_evolution_and_current_architecture.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `00_overview/project_status.md`。
- **替代关系**：覆盖 DEC-036 中 `Extractor / Updater` 的双模块表述；保留 DEC-036
  的 `A_hist` token/cross-attention 参与和 no-bias 约束，也保留 DEC-034 的
  `A_noise`、`H_action=10`、action flow decoder；backbone 口径以后遵循 DEC-041。

### DEC-041 补充（2026-08-14）

- **状态**：`Accepted / Code Updated / Training Pending`。
- **选择**：V1 正式主结构固定为 **single shared WanBlock token stream**：

  ```text
  [Register tokens][Z_obs tokens][Z_future^σ tokens][A_hist/A_cur/A_noise tokens][text tokens]
      -> token type / timestep / position embedding
      -> one shared Wan2.1-style DiT / WanBlock stack
      -> generation head / action flow decoder / Stage2 3D heads
  ```

  不再采用 GigaWorld-style dual-stream / MoT-style backbone；不复制独立 action
  expert transformer；不把 policy/action branch 做成第二套大 backbone。
- **保留项**：
  1. `A_noise` 是 action/policy 主变量；
  2. `H_action = H_nav = 10`；
  3. action output 仍采用 GigaWorld/DreamZero 启发的
     `A_noise -> action flow/diffusion decode -> A_out`；
  4. Stage One 中 action loss = 0，但 action token 格式保留；
  5. Stage Two 是同一视频生成前向上增加 current chunk hidden 的 pose/3D
     supervision；
  6. Stage Three 使用同一 shared WanBlock 的 policy-only 裁剪路径，并通过 replay
     video/3D loss 防止退化。
- **理由**：2026-08-14 的 Stage2 formal 启动暴露出 dual-stream 实现会把当前
  `V1FullWorldNavModel` 参数量推到约 `2.89B`，其中 backbone 约 `2.79B`；而此前
  可训练的 StageOne Wan/latent-prefix 路径约 `1.42B`，峰值约 `39.2GiB`。这说明
  dual-stream 不是“在 StageOne 上加 pose head”，而是实质复制/扩张了 backbone。
  NAV 的目标是共享 Wan video backbone 上同时承载 generation、3D 和 policy，而不是
  引入额外大 action transformer。
- **工程影响**：
  - `src/nav/v1/models/full_model.py` 已将 `V1DualStreamBackbone` 替换为
    `V1SharedWanBackbone`；
  - token packing 改为同一序列：
    `[Register][Z_obs][Z_future^σ][A_noise][text][A_cur]`；
  - causal mask 保证 action/Register/obs 不读 future，`A_cur` 只服务 future
    generation tokens；
  - 静态参数统计从 dual-stream 约 `2.89B` 降至 shared WanBlock 约 `1.217B`，
    backbone 约 `1.109B`；
  - 正式 Stage1/2/3 训练仍需基于该 shared WanBlock 版本重新启动并记录。
- **替代关系**：覆盖 DEC-034 中 “backbone 改为 dual-stream / MoT-style” 的子决策；
  不覆盖 `A_noise`、`H_action=10`、action flow decode、Stage2 3D supervision 等
  最新结论。

### DEC-038 补充（2026-08-14）

- **状态**：`Accepted / Implemented / Smoke Verified`。
- **选择**：当前 NAV 主线可执行代码只保留新版完整模型链路。旧 scaffold、小
  bypass head、旧 A/B Register、旧 `A_query`、旧 action-bias 和旧
  InfiniteWorld adapter 入口从当前代码删除；历史结果保留在 V0 历史章节和 git 历史。
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
- **影响文档**：`AGENT.md`、`04_evaluation/evaluation_reproduction_and_benchmarks.md`、
  `00_overview/project_status.md`、`06_operations/resource_inventory.md`。
- **替代关系**：覆盖 DEC-031/032/033 的旧 `A_query`/Wan patch 实现入口；保留
  其中被 DEC-034/037 吸收的原则，例如 no 20-channel mask、per-token timestep、
  policy-safe attention 和官方 Wan 初始化方向。

### DEC-039 补充（2026-08-14）

- **状态**：`Accepted semantics / diagnostic run invalid for acceptance`。
- **选择**：Stage Two 的 3D supervision 不再默认从 Register mean probe 读取。
  当前正式口径改为从 `C_{t-1}` current/local chunk 的 visual hidden 读取
  frame-level pose/depth：

  ```text
  Register history = C_0 ... C_{t-2}
  Current / Local  = C_{t-1}
  Target future    = C_t

  hidden(C_{t-1})
    -> PoseHead:  [tx,ty,tz,qw,qx,qy,qz,fov_x,fov_y]
    -> DepthHead: depth/depth_conf grid

  Future hidden(C_t noisy)
    -> FutureLatentHead -> video generation loss
  ```

  Register 仍然通过 backbone context 影响 current/future hidden，但不直接承担
  first-version 3D readout。Depth head 与 mask/loss 接口保留；若数据没有可靠
  depth target，则 `depth_mask=0`，不得把零 depth 当作有效监督。
- **理由**：current chunk 保留了最完整的空间 token/grid hidden，更接近
  VGGT/VGGT-Ω 中从当前帧 patch tokens / camera tokens 读出 pose/depth 的方式；
  Register 是历史压缩状态，直接监督 Register 容易把“可导航几何”与“长期记忆”
  混在一起，难以解释。
- **语义实现入口**：

  ```text
  code:
    NAV/src/nav/v1/models/full_model.py
    NAV/src/nav/v1/models/heads.py

  invalid diagnostic, not acceptance evidence:
    NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/
  ```

  说明：上述 diagnostic 缩小了 latent spatial resolution 与 backbone depth，不能
  作为 Stage Two 完整训练、正式 smoke、验收结果或方案有效性证据。后续正式
  Stage Two pose-only 训练必须使用完整模型结构和完整 latent/video 几何。

- **影响文档**：`03_training/training_plan_and_experiment_log.md`、
  `03_training/training_plan_and_experiment_log.md`、
  `00_overview/project_status.md`、`06_operations/resource_inventory.md`。
- **替代关系**：覆盖 DEC-038 smoke 中旧 `geometry_probe(register_after_backbone)`
  作为 Stage Two 主要监督的实现；`geometry_probe` 暂保留为历史兼容/Stage Three
  rehearsal 实验入口，不代表当前 Stage Two 主监督。

### DEC-040 补充（2026-08-14）

- **状态**：`Accepted / Running formal verification`。
- **选择**：Stage Two 第一轮正式验证训练改为 `pose-only`，不启用 depth/point
  supervision。除数据集只使用 RE10K 外，模型结构、latent 几何、loss 接口均按当前
  V1 最终标准执行：

  ```text
  latent:
    [16,4,56,112] T4 micro chunk

  model:
    hidden_dim=1536
    num_backbone_layers=30
    num_heads=12
    register_tokens=128
    H_action=10

  loss:
    L_stage2 = L_visual_flow + 0.1 * L_pose
    depth_loss = 0
  ```

- **启动入口**：

  ```text
  tmux:
    nav_stage2_re10k_poseonly_formal

  run:
    NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/

  doc:
    NAV/doc/03_training/training_plan_and_experiment_log.md
  ```

- **启动前验证**：full-shape data preflight 通过；GPU preflight 在完整模型/完整
  latent/batch=1/grad_accum=1/bf16 下通过，峰值约 39.73 GiB。正式训练使用
  micro_batch=1、grad_accum=16、effective_batch=16。
- **理由**：RE10K 原生提供可靠 camera pose / intrinsic，但不提供 dense GT
  depth。为了先验证 Stage Two 中视频生成与 3D pose supervision 是否能同时起效，
  不再混入 VGGT pseudo-depth 或 MVS 派生 depth，避免把 teacher/pseudo depth 的
  质量问题和主结构问题缠在一起。
- **影响文档**：`03_training/training_plan_and_experiment_log.md`、
  `00_overview/project_status.md`、`06_operations/resource_inventory.md`。
- **替代关系**：此前
  `v1_stage2_re10k_pose_video_1000step_20260814_022646` 已降级为
  `diagnostic invalid for acceptance`，不得作为 Stage Two 验收证据。

### DEC-044（2026-08-17）：Stage2/Stage1 视频生成支路回贴 IW/Wan 原生 branch

- **状态**：`Accepted / implementation diagnostic passed`。
- **选择**：后续正式视频生成与 Stage2 诊断优先使用
  `IWAlignedWorldNavModel`，即 InfiniteWorld/WanModel 原生
  `patch_embedding / time_embedding / Wan blocks / denoise head`。此前自写
  `V1SharedWanBackbone + FutureLatentHead` 版本保留为历史尝试，不再作为
  generation branch 主线。
- **保留项**：Register/action/Stage2 probe 语义仍按 V1 设计维护；
  `A_cur` 通过 Wan cross-attention condition，`A_query/A_noise slot` 通过
  shared action tokens 进入同一 Wan token stream。`return_prefix_video_hidden`
  只用于 probe 读取 hidden，不改变 Wan 主计算。
- **理由**：旧可收敛 StageOne run 加载约 825 个 Wan key；自写 Stage2 branch
  只能加载约 242 个 shape-compatible key，且即使修正 RFlow reversed velocity
  后，visual loss 仍不能快速回到旧量级。IW-aligned branch 已验证可加载 825 key、
  可完整 forward/backward、可接 pose probe。
- **诊断结果**：

  ```text
  rflowfix + V1SharedWanBackbone:
    step10 loss_visual ≈ 1.97

  IWAligned + current prefix + pose:
    6-7s/step, peak ≈ 40.2G, eval loss_visual ≈ 1.25, eval pose ≈ 0.022

  IWAligned strict video-only:
    4.1s/step, peak ≈ 27.9G, eval loss_visual ≈ 1.27
  ```

- **当前解释**：短测 loss 未直接回到旧 fullmix StageOne 的 0.5–0.7 量级，主要
  不是 RFlow 公式、zero text、current full prefix 或 current-register update
  单项导致；下一步应使用旧 fullmix manifest、effective batch=16 和旧 window
  构造做真正可比复现。
- **影响代码**：

  ```text
  NAV/src/nav/v1/models/iw_aligned.py
  NAV/scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py
  Infinite-World/infworld/models/dit_model.py
  ```

- **影响文档**：`03_training/training_plan_and_experiment_log.md`。

### DEC-045（2026-08-17）：旧版快速收敛基线固定为 InfiniteWorld legacy20 branch

- **状态**：`Accepted / reproduced to step20`。
- **选择**：旧版有效基线不再使用当前
  `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World`
  工作树，而使用独立 worktree：

  ```text
  /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20
  ```

  该 worktree 保留旧 InfiniteWorld 20-channel condition-mask video branch，
  并只补最小 `memory_is_precomputed/local_memory` 接口以支持 Register prefix。
- **理由**：历史 checkpoint 证明旧有效 run 的 backbone 是
  `patch_embedding.weight=[1536,20,1,2,2]`，无 `token_type_embedding`，且
  `StageOneActionInterface` 为 820,516 参数的 legacy action probe。当前
  InfiniteWorld 工作树已经改成 16ch/token-type branch，无法复现历史收敛。
- **验证证据**：

  ```text
  run:
    NAV/log/legacy20_stage1_t4_fullmix_convergence_20260817_021207/

  structural audit:
    trainable params = 1,422,300,004
    backbone params = 1,421,390,400
    patch_embedding = [1536,20,1,2,2]
    loaded Wan keys = 825, partial patch expand

  convergence:
    step1  = 1.613
    step10 = 0.815
    step20 = 0.572

  repeat check:
    NAV/log/legacy20_stage1_t4_fullmix_convergence_20260817_024822/
    step1..10 = 1.613, 1.693, 1.223, 1.141, 0.875,
                0.983, 0.891, 0.946, 0.850, 0.815

  historical reference:
    step1  = 1.386
    step10 = 0.712
    step20 = 0.507
  ```

- **影响代码**：

  ```text
  NAV/src/nav/infinite_adapter.py
  NAV/src/nav/stage_one_action_interface.py
  NAV/scripts/train_v1_wan_stage1_t4_history.py
  NAV/scripts/run_v1_wan_stage1_t4_history.sh
  Infinite-World-legacy20/infworld/models/dit_model.py
  ```

- **下一步**：所有最终设定（Stage2 pose/current hidden probe、Stage3 action
  interface、policy-safe mask 等）必须逐项 graft 到该 legacy20 generation
  branch 上，并用 `L_visual` 不退化作为进入下一步的 gate。

### DEC-046（2026-08-17）：Stage2 pose supervision 先采用 legacy20 two-pass bridge

- **状态**：`Accepted as bridge gate / needs held-out eval`。
- **选择**：在 legacy20 有效生成支路上加入 Stage2 pose 监督时，不把
  `z_obs/current chunk` 直接拼入 video generation prefix；改为：

  ```text
  video pass:
    legacy20 Register + local memory + noisy future -> RFlow visual loss

  pose pass:
    clean z_obs/current chunk 作为 prefix
    dummy_future=zeros_like(z_obs), t=0，仅满足 Wan block 非空 noisy segment
    从 prefix_video_hidden 读取 z_obs token
    FramePoseHead(hidden) -> per-frame pose loss
  ```

- **理由**：直接使用 `current_latent_prefix=z_obs` 会改变旧生成支路输入拓扑，
  1-step visual loss 高于严格 legacy video path；two-pass bridge 保留 video
  path 不变，同时让 pose loss 通过同一个 Wan/DiT backbone 反传。
- **验证证据**：

  ```text
  runs:
    log/v1_stage2_legacy20_re10k_pose_video/
      cmp_legacy20_stage2_lambda0_video_gate_20step_20260817_030453/
      cmp_legacy20_stage2_lambda0p1_pose_gate_20step_20260817_030531/

  same seed / same data / same initialization:
    lambda=0:
      visual step1=2.4826, step10=0.4742, step20=0.3510
    lambda=0.1:
      visual step1=2.4826, step10=0.4721, step20=0.3532
      pose   step1=0.6262, step10=0.1174, step20=0.0023

  checkpoint + held-out eval:
    checkpoint:
      log/v1_stage2_legacy20_re10k_pose_video/
        legacy20_stage2_posebridge_save20_20260817_031122/step-000020.pt
    eval:
      result/v1_stage2_legacy20_re10k_pose_video/
        legacy20_stage2_posebridge_save20_eval16_20260817_031439/eval_metrics.json
    metrics:
      visual_velocity_mse = 0.4016
      latent_x0_mse = 0.4103
      baseline_noisy_latent_mse = 1.1025
      pose_mse = 0.0051
      pose_translation_mae = 0.0414
      pose_rotation_angle_deg = 6.7475

  longer stability check:
    checkpoint:
      log/v1_stage2_legacy20_re10k_pose_video/
        legacy20_stage2_posebridge_100step_20260817_031657/step-000100.pt
    train means:
      visual 1-20=0.5456, 21-50=0.3502, 51-80=0.2083, 81-100=0.1850
      pose   1-20=0.1817, 21-50=0.0164, 51-80=0.0054, 81-100=0.0045
    held-out eval32:
      visual_velocity_mse = 0.1691
      latent_x0_mse = 0.2520
      baseline_noisy_latent_mse = 1.0011
      pose_mse = 0.0038
      pose_translation_mae = 0.0477
      pose_rotation_angle_deg = 6.9171
  ```

- **边界**：这是训练链路和不伤 visual branch 的 gate，不是最终 Stage2 完成证据。
  完成 Stage2 仍需要保存 checkpoint，并做 held-out pose eval 与生成质量检查。
- **影响代码**：

  ```text
  NAV/scripts/train_v1_stage2_legacy20_re10k_pose_video.py
  Infinite-World-legacy20/infworld/models/dit_model.py
  ```

### DEC-047（2026-08-17）：legacy20 Stage2 成立，但 action/policy 正式 graft 尚未完成

- **状态**：`Accepted / gap explicitly tracked`。
- **结论**：legacy20 20-channel IW/Register branch 是当前唯一已经复现旧版快速
  generation 收敛的有效支路；two-pass Stage2 pose bridge 已证明在该支路上
  generation 与 pose readout 可以同时收敛。但它尚不满足正式 V1 的全部
  action/policy invariant。
- **具体 gap**：

  ```text
  legacy20 current:
    A_hist = StageOneActionInterface additive latent bias
    A_cur  = InfiniteWorld native action_encoder additive path
    A_out  = legacy action_query/probe side head for checkpoint compatibility
    mask   = 20-channel condition mask + hist/noisy split

  formal V1 required:
    A_hist = independent action tokens into RegisterCell, no additive bias
    A_cur  = DiT condition/context token readable only by future visual tokens
    A_noise -> A_out = action-flow tokens in the shared WanBlock token stream
    policy-safe attention mask: action tokens cannot read future visual tokens
    preferred patch stem = 16 latent channels, no 20-channel mask
  ```

- **理由**：legacy20 Wan self-attention only supports a `num_c` hist/noisy split；
  it does not expose arbitrary token-level policy-safe attention masks. Directly
  appending `A_noise` tokens would either leak future visual tokens to policy or
  break the Wan unpatchify/head assumptions.
- **后续路线**：
  1. `bridge/probe`：保持 legacy20 generation branch 不动，把 policy/action 作为
     register/current hidden adapter 先验证 VLN transfer；必须标为 bridge/ablation。
  2. `formal V1`：实现真正 16ch shared token stream + token mask + action-flow
     decoder；但必须重新通过 generation convergence gate。
- **影响文档**：`03_training/training_plan_and_experiment_log.md`。

### DEC-048（2026-08-17）：在 legacy20 Wan blocks 中加入可选 shared action tail

- **状态**：`Accepted as legacy20-compatible graft / not final 16ch replacement`。
- **选择**：在已经复现旧版收敛的
  `Infinite-World-legacy20/infworld/models/dit_model.py` 中，为 `WanModel.forward`
  增加一个严格可选的 `shared_action_tokens` tail：

  ```text
  old path:
    [visual/register/local/noisy future tokens] -> Wan blocks -> video head

  graft path:
    [visual/register/local/noisy future tokens][A_noise/A_query tail]
      -> same Wan blocks
      -> split:
           visual tokens -> original Wan video head
           action tail hidden -> action decoder / A_out
  ```

- **关键约束**：
  - 不传 `shared_action_tokens` 时，旧版路径保持原样；
  - action tail 是非空间 token，不使用 3D RoPE；
  - action-tail query rows 在 self-attention 中禁止读取 future noisy visual keys，
    防止 policy 侧看见未来视觉；
  - video head 前剥离 action tail，避免破坏 Wan unpatchify；
  - `A_cur` 暂时仍走 InfiniteWorld 原生 `move/view -> action_encoder`
    video-action condition；正式 H=10 policy action 在进入该原生 video condition 前
    必须 pad/裁剪到 IW 期望的 81-frame action 序列，不能直接喂 H=10。

- **验证证据**：

  ```text
  compile:
    py_compile passed:
      Infinite-World-legacy20/infworld/models/dit_model.py
      NAV/src/nav/v1/models/iw_aligned.py

  action-tail full-model 1-step:
    log/v1_stage2_iw_aligned_re10k_pose_video/
      iw_aligned_actiontail_1step_20260817_041636/
    data:
      RE10K real Stage2 sample, latent [16,4,56,112], action_horizon=10
    init:
      Wan2.1 official safetensors, loaded_keys=825
      patch_embedding 16ch -> 20ch partial expansion
    train step1:
      loss_visual = 1.0371
      loss_pose   = 0.8398
      cuda max memory = 39.74 GiB

  no-action-tail regression:
    log/legacy20_stage1_no_actiontail_regression_20260817_041744/
    step1 loss = 1.6130983456969261
    previous legacy20 step1 loss = 1.6130983456969261
  ```

- **边界**：
  - 该 graft 已把 `A_noise/A_query tail` 放进真实 Wan blocks，不再是
    `V1FullWorldNavModel` 的 synthetic/smoke transformer；
  - 但它仍保留 legacy20 的 20-channel mask 和原生 additive `A_cur` video
    condition，因此还不是最终 16ch formal V1；
  - 下一步需要把 action tail 从 learnable query 改成真正
    `A_noise(action_t) -> action flow decoder -> A_out`，并在 Stage3 R2R policy
    数据上验证 action loss / policy-only 推理速度。

- **影响代码**：

  ```text
  Infinite-World-legacy20/infworld/models/dit_model.py
  NAV/src/nav/v1/models/iw_aligned.py
  ```

### DEC-049（2026-08-17）：action tail 升级为 `A_noise(action_t) -> A_out`

- **状态**：`Accepted / full-model real-data chain verified`。
- **选择**：在 DEC-048 的 legacy20 shared action tail 上，正式把 action 输入从
  learnable query fallback 升级为 `A_noise + action_timestep`：

  ```text
  A_noise [B,H_nav=10,6] + action_t
    -> action_noise_in + action position + action timestep embedding
    -> shared legacy20 Wan blocks action tail
    -> action_velocity_head / primitive_head
    -> L_action_flow + CE_aux
  ```

- **训练语义**：
  - Stage One/Two 若提供 `a_noise`，也会以最终格式穿过 shared Wan action tail；
    但 Stage One/Two 默认不计算 action loss；
  - Stage Three policy 使用真实 R2R rendered-policy H=10 action chunk 监督：
    `L_action_flow = MSE(action_velocity, action_target - a_noise)`，
    `CE_aux = cross_entropy(primitive_logits, primitive_id)`；
  - policy-only 验证中仍传入一个 dummy future visual token 以满足 legacy20
    hist/noisy split，但不计算 video loss。

- **验证证据**：

  ```text
  Stage3 real R2R policy chain:
    script:
      NAV/scripts/train_v1_stage3_iw_aligned_r2r_policy.py
    run:
      log/v1_stage3_iw_aligned_r2r_policy/
        stage3_actionflow_r2r_smokelatent_1step_20260817_042532/
    data:
      rendered-policy H=10 manifest
      Wan-VAE encoded rendered R2R obs latent smoke cache
      sample shape:
        history_latents = [4,16,1,56,112]
        z_obs           = [16,1,56,112]
        a_noise/target  = [10,6]
    init:
      Wan2.1 official safetensors, loaded_keys=825
    step1:
      loss_action_flow = 1.3203
      CE_aux           = 2.6719
      total loss       = 1.5859
      seconds/step     = 3.97
      max memory       = 28.07 GiB

  Stage2 video+pose compatibility after A_noise change:
    run:
      log/v1_stage2_iw_aligned_re10k_pose_video/
        iw_aligned_actionnoise_stage2_1step_20260817_042700/
    step1:
      loss_visual = 1.8560
      loss_pose   = 0.4824
      eval loss_visual = 1.1076
  ```

- **边界**：
  - Stage3 这次使用的是 R2R obs latent smoke cache（真实渲染帧 + Wan VAE 编码，
    但只覆盖少量 episode），证明链路，不证明 policy 性能；
  - 下一步需要完成/启动 full R2R obs latent cache，然后做至少数百到数千 step 的
    Stage3 policy training，并监控 action-flow loss 是否稳定下降；
  - `A_hist` 仍未完全 no-bias token 化，`A_cur` 仍是 IW 原生 video condition，
    16ch formal V1 仍需单独 convergence gate。

- **影响代码**：

  ```text
  NAV/src/nav/v1/models/iw_aligned.py
  NAV/scripts/train_v1_stage3_iw_aligned_r2r_policy.py
  ```

### DEC-050（2026-08-17）：Stage3 policy 训练入口加入 video/pose replay gate

- **状态**：`Accepted / full-model chain verified`。
- **选择**：Stage3 训练不再只有 policy loss；正式入口支持可选 RE10K
  video/pose replay：

  ```text
  per optimizer step:
    1) R2R policy batch:
       L_policy = L_action_flow + lambda_ce * CE_aux
       backward()

    2) optional RE10K replay batch:
       L_replay = lambda_video_replay * L_visual
                + lambda_pose_replay  * L_pose
       backward()

    3) optimizer step
  ```

  两次 backward 分开执行，避免同时保留 policy graph 和 video/pose graph 导致显存
  峰值过高。

- **验证证据**：

  ```text
  run:
    log/v1_stage3_iw_aligned_r2r_policy/
      stage3_actionflow_with_replay_1step_20260817_043243/

  data:
    policy: rendered R2R H=10 smoke obs-latent cache
    replay: RE10K T4 latent + pose

  coefficients:
    lambda_video_replay = 0.25
    lambda_pose_replay  = 0.05

  step1:
    loss_action_flow     = 1.2578
    CE_aux               = 2.7656
    loss_video_replay    = 0.7805
    loss_pose_replay     = 0.5234
    loss_replay_weighted = 0.2213
    seconds/step         = 24.74
    max memory           = 39.91 GiB
  ```

- **数据准备状态**：

  ```text
  full R2R obs latent cache started in tmux:
    session: nav_vln_obs_latents_full
    output:
      /sharedata/NAV/derived/v1/vln/obs_latents_r2r_full/
        r2r_standard_20260817_full/
    logs:
      NAV/log/v1_data_prep/obs_latents_r2r_full_shard0_20260817.log
      NAV/log/v1_data_prep/obs_latents_r2r_full_shard1_20260817.log
  ```

- **边界**：当前 replay gate 是 full-model chain gate，不是长训收敛证明。下一步
  需要等 full R2R obs latent cache 可用后，启动多步 Stage3 training，并持续监控
  `loss_action_flow`、`loss_visual_replay`、`loss_pose_replay`。

- **影响代码**：

  ```text
  NAV/scripts/train_v1_stage3_iw_aligned_r2r_policy.py
  NAV/src/nav/v1/models/iw_aligned.py
  NAV/scripts/datasets/encode_v1_stage3_vln_obs_latents.py
  ```

### DEC-051（2026-08-17）：Stage2 采用 strict video pass + separate current-clean pose-probe pass

- **状态**：`Accepted / full-model 1-step verified`。
- **选择**：在 legacy20-compatible graft 中，Stage2 的 video generation loss
  不再因为 pose supervision 而把 `z_obs` 拼进主生成 pass。默认组织改为：

  ```text
  video pass:
    image_cond = Register(history, A_hist tokens)
    local_memory = z_obs[:, :, -1:]
    x = z_future_noisy
    loss_visual = MSE(v_pred, noise - z_future)

  pose-probe pass:
    image_cond = z_obs
    local_memory = z_obs[:, :, -1:]
    x = zeros_like(z_obs)
    t = 0
    read current clean-prefix hidden
    loss_pose = MSE(pose_head(hidden), pose_gt)
  ```

  两个 pass 共享同一个 legacy20 WanBlock/backbone 参数；这不是双 backbone，也不是
  Giga-style dual-stream。它的目的只是让 3D 监督读取 current chunk hidden，同时
  不改变旧版已证明快速收敛的 video branch token layout。

- **主要理由**：
  - 旧版有效生成分支依赖 strict `Register + latest-frame local_memory`；
  - 早先 `include_current_obs_prefix=True` 会改变 video token layout，容易污染
    generation convergence gate；
  - legacy bridge 脚本 `train_v1_stage2_legacy20_re10k_pose_video.py` 已验证
    two-pass 形式可同时让 visual/pose loss 下降；
  - 当前正式 `IWAlignedWorldNavModel` 需要继承该语义，而不是把 pose 需求混进
    video loss pass。

- **验证证据**：

  ```text
  run:
    log/v1_stage2_iw_aligned_re10k_pose_video/
      iw_aligned_twopass_poseprobe_1step_20260817_045819/

  config:
    checkpoint = /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
    Wan loaded_keys = 825
    latent = [16,4,56,112]
    history_steps = 1
    lambda_pose = 0.1
    checkpointing = non-reentrant

  step1:
    train/loss_visual = 1.5098
    train/loss_pose   = 0.5234
    eval/loss_visual  = 1.2806
    eval/loss_pose    = 0.5127
    max memory        = 27.97 GiB
  ```

- **实现修复**：
  - `NAV/src/nav/v1/models/iw_aligned.py`：
    `forward_stage2()` 默认使用 separate current-clean pose-probe pass；
  - `NAV/scripts/train_v1_stage3_iw_aligned_r2r_policy.py`：
    replay 不再因 `lambda_pose_replay>0` 打开 `include_current_obs_prefix`，而是
    继承同一个 strict video pass + separate pose-probe 语义；
  - `Infinite-World-legacy20/infworld/models/dit_model.py`：
    action-tail self-attention 的 visual/action split 改为从当前 `grid_sizes`
    和序列长度推断，避免 gradient checkpoint backward 时被第二个 forward 覆盖
    `num_action_tokens` mutable attribute；
  - `NAV/scripts/train_v1_wan_stage1_t4_history.py`：
    增加默认关闭的 `--disable-checkpoint`，便于防回归短测不落大权重。

- **防回归证据**：

  ```text
  run:
    log/legacy20_ropeinfer_noaction_regression_20260817_050052/

  old Stage1 no-action-tail path:
    step1 loss = 1.6130983456969261
  expected:
    log/legacy20_stage1_t4_fullmix_convergence_20260817_021207/
    step1 loss = 1.6130983456969261
  ```

  因此 action-tail RoPE/checkpoint 修复对无 action-tail 的旧生成路径是 no-op。

- **30-step 收敛 gate**：

  ```text
  cotrain:
    log/v1_stage2_iw_aligned_re10k_pose_video/
      iw_aligned_twopass_cotrain_gate30_20260817_050925/
    lambda_pose = 0.1

  video-only control:
    log/v1_stage2_iw_aligned_re10k_pose_video/
      iw_aligned_strict_videoonly_gate30_20260817_050925/
    lambda_pose = 0.0
  ```

  关键结果：

  ```text
  paired train visual loss over 30 common steps:
    mean(cotrain - videoonly) = +0.00034
    max_abs_diff              = 0.00608

  eval/loss_visual:
    step10 cotrain/videoonly = 0.69540 / 0.69528
    step20 cotrain/videoonly = 0.60807 / 0.60932
    step30 cotrain/videoonly = 0.53856 / 0.53823

  cotrain eval/loss_pose:
    step10 = 0.3496
    step20 = 0.2148
    step30 = 0.0999
  ```

  结论：在当前 RE10K Stage2 gate 中，two-pass pose supervision 没有可见地破坏
  video generation loss 的短程收敛；pose head/共享 Wan hidden 的监督也正常下降。

- **300-step 收敛 gate**：

  ```text
  report:
    NAV/result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json

  paired train visual over 61 common logged steps:
    mean(cotrain - videoonly) = +0.00041
    max_abs_diff              = 0.01373

  last5 train visual mean:
    cotrain/video-only = 0.23593 / 0.23723

  eval/loss_visual:
    step50  cotrain/videoonly = 0.43922 / 0.43414
    step100 cotrain/videoonly = 0.35400 / 0.35079
    step150 cotrain/videoonly = 0.31832 / 0.31861
    step200 cotrain/videoonly = 0.24091 / 0.23924
    step250 cotrain/videoonly = 0.23319 / 0.23505
    step300 cotrain/videoonly = 0.41155 / 0.41321

  cotrain eval/loss_pose:
    step50  = 0.03918
    step100 = 0.01973
    step150 = 0.00329
    step200 = 0.00232
    step250 = 0.01880
    step300 = 0.00371
  ```

  结论：300-step gate 通过。当前 two-pass legacy20-compatible graft 在
  RE10K-only Stage2 设置下，pose supervision 没有破坏 video generation
  branch 收敛；pose probe 也能从共享 Wan hidden 中学到有效 pose signal。

- **旧版命令复跑证据**：

  ```text
  run:
    log/legacy20_stage1_t4_fullmix_cmdcheck_gpu1_after_videoonly_20260817_054759/

  observed:
    step1  = 1.6130983456969261
    step10 = 0.8149177767336369
    step20 = 0.5725867711007595

  baseline:
    log/legacy20_stage1_t4_fullmix_convergence_20260817_021207/
    step1  = 1.6130983456969261
    step10 = 0.8151123039424419
    step20 = 0.5724489018321037
  ```

  因此 legacy20 Stage1 命令、数据根、官方 Wan2.1-1.3B 初始权重、20-channel
  patch stem、Register/history 构造和 text cache 路径仍可复现旧版快速收敛。

- **再次复跑证据（2026-08-17 07:21，有效环境显式锁定 legacy20）**：

  ```text
  run:
    log/legacy20_stage1_t4_fullmix_cmdcheck_rerun_20260817_072137/

  env:
    NAV_INF_WORLD_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World-legacy20

  observed:
    step1  = 1.6130983456969261
    step5  = 0.8757058940827847
    step10 = 0.8148440010845661
  ```

  说明有效旧版分支不仅在之前 GPU1 run 中可复现，也可以在当前工作树通过
  显式 legacy20 环境复现到 step10。直接调用脚本若未设置
  `NAV_INF_WORLD_ROOT` 会默认指向新版 `Infinite-World` 并触发 16/20-channel
  mismatch；后续所有复现/训练命令必须显式设置 legacy20 root 或使用包装脚本。

- **Stage3 replay 回归证据**：

  ```text
  run:
    log/v1_stage3_iw_aligned_r2r_policy/
      stage3_twopass_replay_regression_1step_20260817_050513/

  policy:
    real rendered R2R smoke obs latents
    A_hist = [4,10], A_noise/A_out = [10,6]

  replay:
    RE10K T4 latent + pose
    lambda_video_replay = 0.25
    lambda_pose_replay  = 0.05

  step1:
    loss_action_flow  = 2.3281
    CE_aux            = 2.7813
    loss_video_replay = 1.0281
    loss_pose_replay  = 0.5508
    max memory        = 28.12 GiB
  ```

- **Stage2/Stage3 入口修复**：

  `NAV/scripts/train_v1_stage2_iw_aligned_re10k_pose_video.py` 和
  `NAV/scripts/train_v1_stage3_iw_aligned_r2r_policy.py` 现显式把以下路径加入
  `sys.path`：

  ```text
  NAV/src
  NAV/scripts
  ${NAV_INF_WORLD_ROOT:-../Infinite-World-legacy20}
  ```

  这样 Stage2/Stage3 正式入口不再依赖调用者手工设置 `PYTHONPATH` 才能找到
  `infworld` 或相邻 helper。已通过 `py_compile`、Stage2 data-only preflight，
  以及真实 full-cache manifest dataset 构造检查：

  ```text
  history_latents = [4,16,1,56,112]
  A_hist          = [4,10]
  z_obs           = [16,1,56,112]
  A_noise/target  = [10,6]
  ```

  该检查只验证入口与数据形状，不作为 policy 性能证据。

- **Stage2 step500 decoded eval 证据（2026-08-17 08:18）**：

  `iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532` 已到 step500，并
  保存：

  ```text
  log/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_twopass_cotrain_re10kfull_2k_20260817_063532/
      checkpoints/step_000500.pt
  ```

  训练内指标：

  ```text
  step500 train/loss_visual = 0.10883779078722
  step500 train/loss_pose   = 0.000949859619140625
  eval@500 visual           = 0.2380673922598362
  eval@500 pose             = 0.003208160400390625
  ```

  手动补跑 decoded eval 后输出：

  ```text
  result/v1_stage2_iw_aligned_re10k_pose_video/
    iw_aligned_re10kfull_2k_step500_eval/
  ```

  关键指标：

  ```text
  velocity MSE       = 0.15447463025338948
  latent x0 RMSE     = 0.27066083753015846
  latent x0 cosine   = 0.931394575163722
  pose trans MAE     = 0.034744832722935826
  pose rotation deg  = 7.362439222633839
  RGB PSNR           = 21.63981533050537
  ```

  4 组 `obs/gt_future/pred_future_proxy` mp4 均可播放，格式为
  `896x448 / 13 frames / 12fps`，pred mean/std 与 GT 接近且 `motion_l1`
  非零。当前判断：Stage2 two-pass graft 至少到 step500 时已给出
  “generation loss + pose probe + decoded proxy video” 三重可用证据。
  这仍不是最终 Stage2 完成声明，后续继续等待 step1000/1500/2000 eval。

  同时修复 `scripts/eval_v1_stage2_iw_aligned_re10k_pose_video.py`：eval
  加载 checkpoint 时先 `remove_hmpc()`，并只允许未使用的
  `backbone.latent_encoder.*` missing，避免 HPMC 残留导致 checkpoint 误报错。
  之后又加固 Stage2/Stage3 新 checkpoint 元信息：

  ```text
  runtime_flags:
    hmpc_removed
    backbone_use_convenc
  ```

  evaluator 对旧 checkpoint 默认按 IW-aligned 训练态 `remove_hmpc()`，对未来
  带 `runtime_flags` 的 checkpoint 按 flag 恢复；旧 `step_000500.pt` 已通过
  CPU 加载回归。


新增决策时写明日期、状态、选择、主要理由、影响文档和替代关系。只有已经确认并
落实到设计或配置的事项标记为 `Accepted`；讨论中的选项使用 `Proposed`。
### DEC-047：Stage2 step3000 后降低学习率续训

- 时间：2026-08-25
- 决定：停止原 `step2600 -> step3600, lr=1e-5` 续训任务，改从已落盘的 `step3000` 恢复。
- 新学习率：`2e-6`。
- 优化器：保留 checkpoint 中 AdamW 的一、二阶动量，但在恢复后显式覆盖所有 parameter group 的学习率；恢复前后学习率必须写入 `resume_audit.json`，避免 checkpoint 内旧 LR 静默覆盖命令行设置。
- Stage3 线性动作头训练不受本次调整影响。

### DEC-048：Stage2/Stage3 使用同一视频与 3D 评测链路

- 时间：2026-08-25
- 决定：正式 Stage2 evaluator 同时识别 Stage2 checkpoint 的 `data_config` 与 Stage3 checkpoint 的 `stage2_replay_data_config`。
- 目的：对 Stage2 与 Stage3 使用完全相同的数据构造、完整 RFlow 去噪和 pose 指标口径，直接衡量 Stage3 policy cotrain 后的生成/3D 遗忘，而不复制或改写 checkpoint。
- 配对随机性：模型 checkpoint 加载完成后重新设置评测 seed，消除 Stage2/Stage3 不同 action head 构造过程对 RNG 的消耗差异；两版必须使用相同窗口、timestep 与初始 diffusion noise。
