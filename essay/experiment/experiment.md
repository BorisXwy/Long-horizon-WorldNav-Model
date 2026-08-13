# 实验

本节预留 NAV 的完整实验结构。当前版本是论文实验章节草稿，而不是已完成结果汇总；除已明确标注的 Stage One 运行链路外，实验结论均为【未验证】。后续每个实验完成后，应将表格、图、指标和日志路径回填到对应位置。

## 4.1 实验目标

实验围绕本文核心命题展开：spatial-aware streaming world model 中的 quasi-infinite memory 是否可以作为 embodied observation history。我们需要验证四个问题：

1. **Memory 是否有效。** Spatial Register Memory 是否比无长期记忆、latest local memory 或 HPMC 更能利用长历史？
2. **Memory 是否 spatial-readable。** Register-after-DiT 表示是否能被读出 depth、pose、point/correspondence 等空间结构？
3. **Memory 是否有助于 navigation。** 将 \(M_t\) 或 \(F_t^{nav}\) 输入 VLN policy 是否提高 SR、SPL 和长程任务表现？
4. **Action-centered interface 是否必要。** 训练时使用 future visual/3D consequence、推理时关闭 generation branch 的设计是否兼顾性能和效率？

## 4.2 数据集与评测协议

### Video/world-model pretraining data

Stage One 使用 DL3DV、SpatialVID、RE10K 和 Argoverse2 等视频数据构造 T4 micro latent windows。每个 micro chunk 包含 13 个 RGB frames，stride 为 12，经 Wan VAE 独立编码为 \([16,4,H,W]\) latent。history span 以 IW-equivalent chunks 表示：IW-1/4/8/16 分别对应 7/27/54/107 个 T4 micro history updates。

预留文字：我们将在正式版本中报告各数据集 episode 数、window 数、history bucket 分布和 sampler ratio。当前 Stage One manifest 与训练入口已实现【已验证→`doc/04_training/v1_stage_one_t4_iw_aligned.md`】。

### Geometry supervision data

Stage Two 使用带 pose/intrinsics 或高置信 pseudo geometry 的视频样本，构造 depth、relative pose、pointmap/correspondence 和 confidence mask。pose 和 intrinsics 只作为监督来源，不作为推理输入。

预留文字：正式版本将报告 geometry label 来源、confidence filtering、metric/normalized scale 处理，以及每个数据集在 Stage Two 中的权重【未验证：Stage Two 尚未实现】。

### VLN policy data

Stage Three 使用 R2R-CE、RxR-CE、ScaleVLN 和 LHPR-VLN 等导航数据。每个样本包含 instruction、当前观测、历史 observation/action prefix 和未来 \(H=4\) 步 action labels。推理采用闭环 first-action execution：模型预测 action chunk，但每步只执行第一个动作，随后接收新观测并更新 Register。

预留文字：正式版本将报告 train/val_seen/val_unseen split、action vocabulary、STOP padding/masking、evaluation environment 和 episode-level metrics【未验证：Stage Three 尚未实现】。

## 4.3 Baselines

我们预留以下 baseline，用于分别验证 memory、geometry 和 action-centered interface 的贡献。

| 类别 | Baseline | 目的 | 状态 |
| --- | --- | --- | --- |
| No long memory | latest local only | 检查长期 Register 是否必要 | 未验证 |
| Zero/Register ablation | zero Register / shuffled history | 检查模型是否真实读取历史 | 未验证 |
| Streaming video memory | InfiniteWorld HPMC | 对比递归 Register 与启发式压缩 memory | 未验证 |
| Growing cache | rolling KV cache / full history tokens | 对比固定预算与增长型历史 | 未验证 |
| Geometry ablation | no 3D supervision | 检查 spatial readability 是否必要 | 未验证 |
| Policy interface | policy-only vs video-generation inference | 检查推理时关闭 generation branch 的效率与性能 | 未验证 |
| VLN baselines | StreamVLN / NavFoM / strong VLN policy | 对比标准导航性能 | 待接入 |

## 4.4 Main Results: Navigation Performance

主实验评估 NAV 在 VLN benchmark 上的导航性能。核心指标包括 Success Rate (SR)、Success weighted by Path Length (SPL)、Navigation Error (NE)、Oracle Success Rate (OSR) 和 Stop accuracy。

预留表格：

| Method | Memory | Geometry sup. | Policy inference | Val Seen SR | Val Seen SPL | Val Unseen SR | Val Unseen SPL |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| VLN baseline | task memory | no | policy-only | TBD | TBD | TBD | TBD |
| Local-only NAV | latest local | optional | policy-only | TBD | TBD | TBD | TBD |
| NAV w/o 3D | Register | no | policy-only | TBD | TBD | TBD | TBD |
| NAV | Spatial Register | yes | policy-only | TBD | TBD | TBD | TBD |

预留结果文字：若 NAV 在 val-unseen split 上显著优于 local-only 和 no-3D variants，则说明 streaming world-model memory 不仅能压缩历史，还能提供可泛化的环境状态【未验证】。

## 4.5 Long-Horizon Memory Evaluation

该实验验证 quasi-infinite memory 接口是否真的能利用长历史。我们按 IW-1/4/8/16 history span 构造 evaluation buckets，并比较 normal Register、zero Register、shuffled history、latest local only 和 HPMC。

预留表格：

| History span | Local only | Zero Register | Shuffled history | HPMC | NAV Register |
| --- | ---: | ---: | ---: | ---: | ---: |
| IW-1 | TBD | TBD | TBD | TBD | TBD |
| IW-4 | TBD | TBD | TBD | TBD | TBD |
| IW-8 | TBD | TBD | TBD | TBD | TBD |
| IW-16 | TBD | TBD | TBD | TBD | TBD |

预留结果文字：如果 NAV Register 在长 history span 上相对 local-only 和 shuffled-history 的优势扩大，并在固定预算下保持更好的性能/成本比，则支持 quasi-infinite memory 作为 long-horizon observation history 的主张【未验证】。

## 4.6 Spatial Readability and Layer Selection

该实验回答两个问题：Register 是否编码可读的空间结构，以及哪一层最适合导航读取。我们冻结 Stage Two backbone，在不同 DiT 层和 Register-after-DiT 表示上训练轻量 geometry probes，评估 depth AbsRel、relative pose error、point/correspondence error，并与 navigation validation performance 对齐。

预留表格：

| Readout layer | Depth AbsRel | Pose error | Point error | Val Unseen SR | Val Unseen SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25% layer | TBD | TBD | TBD | TBD | TBD |
| 50% layer | TBD | TBD | TBD | TBD | TBD |
| 75% layer | TBD | TBD | TBD | TBD | TBD |
| final layer | TBD | TBD | TBD | TBD | TBD |
| Register-after-DiT | TBD | TBD | TBD | TBD | TBD |

预留结果文字：若最优 navigation layer 与较低 geometry probe error 对齐，则说明 spatial readability 与 navigation usefulness 相关；若二者不完全一致，则需要报告两者差异，并以 navigation validation performance 选择 \(l^\*\)【未验证】。

## 4.7 Ablations

### 3D supervision

比较 Stage Two with/without geometry supervision。指标包括导航 SR/SPL、geometry probe error 和 generation loss。

预留结论文字：如果去掉 3D supervision 后 novel-scene navigation 或 revisit task 明显退化，则说明 geometry readability 是 embodied observation history 的必要属性，而非辅助正则【未验证】。

### Action-centered interface

比较三种设置：正确区分 \(A^{hist}, A^{cur}, A^{query}\)；错误地将 current action 作为 policy condition；去掉 action-conditioned visual consequence。该实验用于验证因果接口是否避免 leakage 并保留 action-dependent dynamics。

预留结论文字：若泄漏设置训练指标虚高但闭环性能下降，且去掉 \(A^{cur}\) 后 future consequence prediction 退化，则支持 action-centered interface 的必要性【未验证】。

### Replay during Stage Three

比较 policy-only fine-tuning 与加入 video/3D replay 的 Stage Three。观察导航性能、geometry probe 保持情况和 generation loss 漂移。

预留结论文字：若 replay 能维持 geometry/readout 能力并减少 shared backbone forgetting，则说明 Stage Three 中 replay loss 是维护 world representation 的关键【未验证】。

## 4.8 Efficiency

该实验评估 NAV 的固定预算 memory 和 policy-only inference 效率。指标包括 memory token 数、per-step latency、GPU memory、是否执行 generation branch，以及不同 history span 下的成本增长。

预留表格：

| Method | Explicit memory grows with horizon? | Generation at inference? | Latency | Memory |
| --- | --- | --- | ---: | ---: |
| Full history tokens | yes | no | TBD | TBD |
| Rolling KV cache | yes / windowed | optional | TBD | TBD |
| HPMC | fixed compressed | yes for generation | TBD | TBD |
| NAV | fixed Register | no | TBD | TBD |

预留结果文字：NAV 的理想结果是随着 history span 增长，policy 读取成本保持近似恒定；推理关闭 generation branch 后延迟显著低于 imagine-then-act variants【未验证】。

## 4.9 当前已落地状态

截至当前草稿，已落地内容主要是 Stage One 的正式训练链路：

- official `Wan2.1-T2V-1.3B` safetensors initialization；
- HPMC removed / bypassed；
- `latent_prefix` Spatial Register Memory；
- T4 micro chunk，target latent \(T=4\)；
- IW-1/4/8/16 等价 history window 混训；
- action interface format path，`lambda_action_format=0`；
- 运行记录：`log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`，截至本次检查记录到 step 680，未见 complete 事件。

Stage Two、Stage Three、导航主结果、geometry probe 和 efficiency ablation 均尚未完成，不能在论文中以已证口吻表述。
