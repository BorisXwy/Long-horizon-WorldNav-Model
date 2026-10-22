# NAV 研究总览、核心 Idea 与执行计划

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-021` |
| 类型 | 研究问题、相关工作、方案路线与实验计划总览 |
| 状态 | Live / Research Roadmap |
| 更新时间 | 2026-10-22 |
| 职责 | 统一说明 NAV 的研究动机、相关工作启发、已形成的结论、当前问题、V1 方案、已有尝试、多卡复现实验和新数据/训练计划。 |

## 当前入口结论

NAV 要解决的问题是：视频世界模型能够利用长视频历史，但它通常以视频生成作为终点；VLN policy 能够根据当前 observation 和 instruction 输出动作，但历史通常以短窗口、KV cache 或任务专用状态保存。我们希望把两者合成一个可流式更新的 embodied world representation：

```text
真实观测与已执行动作
  -> Register recurrent update
  -> 固定数量的历史 Register tokens
  -> shared WanBlock
       ├─ future video consequence prediction
       ├─ current-chunk 3D / pose readout
       └─ navigation action prediction
```

核心假设是：长期历史不需要以原始视频 token 保留；只要 Register 能持续吸收历史视觉和历史动作，并且 shared Wan hidden 具备可读的空间结构，policy 就可以在固定输入预算下使用长程 observation history。这个假设目前有结构和短程训练证据，但还没有完成最终的长历史视频、3D 和导航联合验收，仍标记为【未验证】。

当前研究主线是 V1：`Wan2.1-T2V-1.3B official init + T_latent=4 + Register + action-centered interface + single shared WanBlock + Stage1/2/3`。GigaNav、InfiniteWorld、DreamZero、GigaWorld、StreamVLN 和 BridgeVLA++ 用来提供结构来源、复现对照和消融证据，不替代 V1 正式主线。

## 1. 研究问题与论文叙事

### 1.1 问题定义

导航智能体每次只接收当前 observation，但其决策往往依赖较早的场景布局、已经走过的路径、动作后果和 instruction。直接把所有历史帧输入 DiT 会带来三个问题：token 数随时间增长、视频生成的高维 latent 和 policy 的小动作输入不匹配、历史压缩结果是否包含空间信息无法解释。

因此我们把问题拆成三个可检验的命题：

1. **Memory 命题**：固定预算的 RegisterCell 是否能替代不断增长的原始历史，并在 IW-1/4/8/16 及更长历史下保持预测能力【未验证】。
2. **Spatial 命题**：Register 和 shared Wan hidden 是否能从 current chunk 中恢复 frame-level pose 等空间结构【未验证】。
3. **Embodied 命题**：同一表征能否在训练时服务 future video consequence prediction，在推理时关闭生成支路后服务 VLN action prediction【未验证】。

论文故事应按“可解释的 streaming world representation”展开：Stage1 学 predictive memory，Stage2 让 memory/hidden 具备空间可读性，Stage3 将同一表征接入闭环 policy。导航指标是第三阶段的验证，不是唯一创新点；视频生成和空间读出必须有独立、可复现的质量证据。

### 1.2 预期贡献

预期贡献由以下三部分组成，均需通过正式代码、日志和结果产物支持后才能进入论文主结论：

- 一个固定 token 预算、可在线递归更新的 Register interface，把历史视觉和历史动作压缩成 policy 可读取的 memory。
- 一个 single shared WanBlock token stream，通过 action-centered mask 同时支持 video generation、3D readout 和 policy action，而不让 future visual target 泄漏到 action branch。
- 一个三阶段训练和评测协议，使 video generation、spatial readability、navigation policy 的关系可以分别测量和共同验证。

## 2. 相关工作与得到的启发

### 2.1 InfiniteWorld、LingBotWorld 与长视频世界模型

InfiniteWorld 证明了 Wan2.1 类视频 DiT 可以通过 HPMC 和 local memory 自回归生成很长的视频。其关键思路是每轮只保留有限的局部条件，并把旧历史压缩后用于下一 chunk；模型规模本身不是主要瓶颈，真正的开销来自高维 video latent、较长条件和多步 denoising。我们已经完成 InfiniteWorld 官方权重、VBench 四指标、3-chunk 以及长程协议的本地复现记录【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】。

LingBotWorld、FantasyWorld 等工作进一步说明，长视频训练通常依赖固定 chunk、teacher-forcing 或 autoregressive rollout 的组合；训练样本不是把整段视频一次性塞进 transformer，而是从 episode 中切出连续窗口，使用已知历史预测下一个 chunk。对 NAV 的启发是：episode 可以先切成可复用的 T4 latent，history 长度和窗口位置在训练时采样，Register rollout 也应在线完成。

### 2.2 VGGT / VGGT-Ω 与 Register 表征

VGGT 系列的关键启发是使用 token-level latent 表征整合跨帧空间信息，并通过轻量 probe/task head 从 hidden state 恢复 camera、depth、pointmap 等几何量。VGGT-Ω 中 learnable register 的价值不在于保存一个固定场景，而在于提供持续读写的全局槽位。

这启发了 NAV 的两点设计：第一，R0 不应依赖某条训练视频，正式路径使用固定的 `R_null`，后续状态由 visual chunk 和 `A_hist` 递归更新；第二，3D 监督应从 current-clean hidden/Register 中读取，而不是对生成 future 的不稳定结果做监督。当前 probe 层选择和 pose 读出仍未完成收敛验证。

### 2.3 DreamZero、GigaWorld 与共享 backbone 的 action/video 训练

DreamZero 和 GigaWorld 展示了 Wan backbone 可以同时承载视频和动作变量：video 分支处理 noisy future visual latent，policy 分支使用较少的 action token，通过 mask 保持因果边界；推理 policy 时可以关闭 video generation branch。GigaWorld 的重点不是让 policy 读完整未来视频，而是把 action 作为小输入接口放入共享 transformer。

我们已经复现 GigaWorld transformer-only 入口，并构建了使用 Wan2.1-1.3B 的 GigaNav policy-only/cotrain 消融【已验证→`doc/01_model/giga_nav_ablation_wan21_13b.md`】。GigaNav 的结果说明 Wan backbone 在 R2R action supervision 下可以有效降低 action loss，但它没有 Register 长历史和 Stage2 3D 监督，因此只能证明“backbone/action 接口可训练”，不能证明 NAV 主线已经有效。

### 2.4 StreamVLN、DualVLN、WNM3D 与导航侧流式表示

StreamVLN 和 DualVLN 强调在线 observation、有限上下文和闭环反馈；LogiPlanner 等工作把场景状态、当前 observation 和 instruction 分开处理，再输出导航动作。它们共同说明 policy 的有效输入应当是压缩后的 state/history、当前局部 observation 和 instruction，而不是不断增长的原始视频。

我们已按官方入口完成 StreamVLN 的 R2R 子集开环和闭环复现【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】。这些复现给出的主要启发是：闭环执行协议、observation 时间对齐和 action label 构造会显著影响结果；导航模型的 action head 往往比视频生成模型简单，但必须确保样本中的当前 observation、未来 action 和 STOP 语义严格对齐。

### 2.5 相关工作的共同结论

相关工作共同支持以下工程判断：

- 视频生成模型适合学习空间和动态表征，但输入输出 token 预算太大，不能直接作为实时 policy 接口。
- policy 模型需要小而稳定的 action readout；future video 可以作为训练信号，不能作为 policy 的必要输入。
- 历史压缩必须有显式的读写语义，否则很难判断模型是否真的使用了长程历史。
- 3D supervision 应该约束 current observation 的 hidden 表征，不能只看 video loss 是否下降。

## 3. 已形成的结论与证据边界

### 3.1 已形成的结论

| 结论 | 证据与边界 |
| --- | --- |
| `T_latent=4` micro chunk 适合统一视频和导航数据接口 | 已完成 T4 latent 编码、R2R train cache 与多种 loader 读取；见 `doc/02_data/data_preparation_schema_and_status.md`。它解决了数据接口问题，长程有效性仍需验证。 |
| Register 应承担 long history，`Z_obs` 保留当前局部观测 | 已写入 V1 标准并在 IWAligned/正式模型接口中实现；长历史收益尚未完成正式 gate。 |
| A_hist、A_cur、A_noise 需要区分 | mask 和数据 schema 已实现/记录；action leakage 与无 action 条件的系统 ablation 尚未完整跑完。 |
| single shared WanBlock 是正式主线 | DEC-041 已固定；历史 dual-stream/MoT 只保留为 GigaWorld related work 或 ablation。 |
| GigaNav 的 Wan action path 可以收敛 | GigaNav policy-only/cotrain 已完成 5,000 steps，action loss 明显下降；其结果不能外推到 Register、3D 或生成质量。 |
| 旧 InfiniteWorld 对齐生成支路可以复现 | legacy20 IWAligned Stage1 和短程 Stage2 gate 已有 visual/pose 结果；这些结果不等于最终 V1 全标准验收。 |
| 训练数据应先统一切 chunk/latent，再在线抽 history window | R2R 与视频 T4 cache 已按此规则组织；Register 不预存。 |

### 3.2 尚未解决的核心问题

1. **V1 的 3D 监督还没有形成可靠证据。** 过去 pose loss 能下降，但 probe 预测和生成质量没有同时达到稳定标准；需要先解决 hidden readout 层、pose target 对齐和 loss 量级。
2. **长历史训练的梯度和样本语义仍需确认。** IW16 等价 history 会包含很多 Register update，必须采用完整 forward、合理的 truncated BPTT 或中间状态梯度策略，不能把一条长 rollout 误当成一个简单大 batch。
3. **视频数据和室内导航动作分布不同。** RE10K、DL3DV、SpatialVID、Argoverse2 的 action 是 pose-derived camera motion，R2R/RxR 的 action 是 simulator expert path；视频数据适合作为 `a_condition`，不能直接视为 policy label。
4. **V1 生成支路尚未达到 InfiniteWorld 对齐的稳定性。** 旧版生成 loss 可以下降，但完整新版 token/mask/Register/3D 组合后的解码质量和长程 autoreg 还没有通过同协议对照。
5. **评测尚未形成一条完整闭环。** InfiniteWorld/VBench 和 StreamVLN 子集已复现，V1 的统一 iWorldBench、长历史 memory ablation、3D probe 与 R2R/RxR policy 指标尚未全部闭合。

## 4. 当前 Idea：Action-centered Spatial Register WorldNav

### 4.1 状态更新

每个 episode 在数据侧先切成 `T_latent=4` 的 micro chunk。设当前预测边界为：

```text
C_hist = [C_0, ..., C_{t-2}]   历史 chunk
Z_obs  = C_{t-1}                当前局部观测
Z_future = C_t                  下一 chunk 生成目标
```

`C_hist + A_hist` 依次进入 RegisterCell，得到固定数量的 `R_t`；`Z_obs` 不再重复写入 history。每个真实 observation 到来后，模型用新的 `A_hist` 和 visual chunk 更新 Register，policy 只读取 `R_t + Z_obs + instruction`。

### 4.2 共享 backbone 与 token 角色

正式主干是单一 shared WanBlock token stream：

```text
main tokens:
  Register R_t
  Z_obs
  noisy Z_future
  A_noise
  text / empty text
  A_cur

outputs:
  future visual velocity -> video generation loss
  current hidden/Register probe -> pose/3D loss
  A_noise hidden -> A_out / combo logits -> policy loss
```

三类 action 变量的语义固定为：

- `A_hist`：历史已执行 action 或 pose-derived motion，只用于 Register update。
- `A_cur`：当前 observation 到 future chunk 的动作条件，只服务 video generation consequence prediction。
- `A_noise/A_out`：policy branch 的 noisy action slot 和预测输出，Stage3 使用 VLN label 监督。

当前正式 action schema 使用 `combo_id = trans_id * 12 + rot_id`，policy 输出 `combo_logits [B,H_action,144]`；action horizon 由 backbone 保留的固定槽位承载，具体只读出的 horizon 必须在实验记录中明确。

### 4.3 Mask 关系

```text
A_noise / policy tokens:
  可读 Register、Z_obs、text、自身 action tokens、action timestep
  不可读 Z_future、A_cur、future clean target 和 video noise

Z_future / video tokens:
  可读 Register、Z_obs、text、A_cur 和自身 noisy future tokens
  不读取 policy target，避免 policy 条件被视频分支污染

A_cur:
  只作为 future visual 的 condition
  不进入 Register 或 policy action readout
```

这套 mask 让同一个 backbone 同时学习“动作导致什么视觉后果”和“当前历史/观测下该执行什么动作”，又避免把未来视频或 clean action 泄漏给 policy。该关系已在正式结构文档和 mask 代码中记录；完整性能收益尚未验证。

## 5. 预计做法：三阶段训练

### Stage1：Predictive memory pretraining

使用 RE10K、DL3DV、SpatialVID 和 Argoverse2 的视频 episode。所有 episode 先离线切 T4 latent，训练时随机采样 history window，history bucket 混合 IW-1/4/8/16 等价跨度。

样本输入为 `C_hist、Z_obs、Z_future、A_hist、A_cur、text`，没有 text 的视频使用 empty embedding。Register update 和 A_cur condition 都保留一定比例全零 action，覆盖无 action/无条件场景。Stage1 只优化 visual RFlow/diffusion loss，但保留 A_noise/A_out token 位置和 mask，让后续 Stage2/3 读取同一结构。

第一阶段的验收是：训练 loss 能稳定下降；不同 history bucket 的 loss 不出现系统性崩坏；single-chunk 解码、teacher-forced 多 chunk 和 autoregressive rollout 的视频质量都能复现；在 VBench/iWorldBench 的同口径指标上不明显低于 InfiniteWorld baseline【未验证】。

### Stage2：Video generation + 3D supervision

Stage2 从 Stage1 checkpoint 继续，在相同 video forward 上增加 current chunk 的 frame-level pose supervision。先做 probe sweep：冻结主干，分别从中间层、后层、final layer、Register pooled state 和 current hidden grid 读取 pose head，比较 pose error 与 visual loss 变化；选出可读层后再做 shared backbone + 3D loss 的正式训练。

Stage2 的训练目标为：

```text
L_stage2 = L_visual + λ_pose * L_pose
```

有可靠 depth/point target 时再增加对应 loss；在当前数据条件下，pose 是优先的 3D supervision。3D target 只作为 teacher signal，不进入 video/policy 输入。验收要求是 pose probe 明显优于 no-3D 对照，同时 visual generation loss 和解码质量不退化【未验证】。

### Stage3：VLN policy + replay

Stage3 使用 R2R train 作为主导航数据，RxR 作为 multilingual/generalization 扩展。policy 样本包含完整 episode prefix、在线 Register rollout、当前 `Z_obs`、instruction 和 action label；只预测未来 action chunk，闭环时执行第一个 action，再接收下一 observation 更新 Register。

Stage3 loss 为：

```text
L_stage3 = λ_action * L_combo_CE
         + λ_visual_replay * L_visual
         + λ_3d_replay * L_pose
```

action label 使用 R2R/RxR expert action，STOP 按吸收态处理；终止后可以复制 observation 完成存储 padding，但 sampler 只保留一个 terminal window。Replay 的目的不是提高视频指标，而是防止 policy fine-tuning 破坏 predictive memory 和 spatial readout。验收必须同时报告 action CE、majority baseline、open-loop action accuracy、R2R/RxR closed-loop SR/SPL 和 video/pose replay 漂移【未验证】。

## 6. 已有尝试与当前判断

| 尝试 | 得到的结果 | 当前定位 |
| --- | --- | --- |
| InfiniteWorld A/B、HPMC replacement、RE10K 短程训练 | A/B video loss 可下降；VBench 小规模对照接近 InfiniteWorld，A 在部分指标更好 | V0 baseline 与 generation sanity check，不能代表 V1 |
| IWAligned legacy20 Stage1 | 旧 Wan/InfiniteWorld 生成支路收敛被复现 | 已验证旧接口，作为新结构回归基线 |
| IWAligned Stage2 video + pose gate | visual 与 pose proxy 同时下降，短程 video loss 未明显恶化 | 已验证 gate；不是完整 V1 3D 成功结论 |
| V1 full pipeline smoke | Stage1/2/3 变量、梯度更新、videogen/policy 两种推理链路可跑通 | 链路验证；不等价于训练质量验收 |
| Stage2 video-only / pose cotrain / linear probe | visual loss 接近，pose probe 仍不稳定，尚未证明 3D 监督有效改变表征 | 说明当前问题在 probe/数据/监督路径，需重新设计实验 |
| Stage3 R2R H1/H4 及自然/均衡采样 | loss 震荡、容易落到 MOVE/STOP majority，未形成稳定 navigation learning | 数据和 label 对齐问题尚未完全解决 |
| GigaNav policy-only/cotrain | Wan2.1-1.3B 在 R2R action supervision 下 action loss 能明显下降，5k cotrain 已完成 | 独立 action 可训练性 ablation；不含 Register/3D 长历史 |
| StreamVLN 官方闭环复现 | 官方模型和 Habitat 评测入口跑通，结果与子集协议一致 | 导航评测工程基线 |
| GigaWorld/Fast-WAM transformer smoke | Wan/WAM action/video token 组织和 policy-only 推理路径得到代码级复现 | 结构启发和 latency 对照 |

结论是：目前最可靠的正面证据来自旧版生成支路和 GigaNav action 分支；最关键的 V1 主张，即“同一 Register/shared backbone 同时具备长历史、空间可读和导航作用”，还没有闭合证据链。

## 7. 多卡资源到位后的重做与复现实验

多卡资源到位后，先建立可重复的 baseline，再逐项加入 Register、3D 和 replay。所有实验都使用正式 Wan backbone、正式 dataloader、正式 loss 和完整 inference/eval entrypoint；只缩小数据量、步数或 micro batch 时必须在日志中标注 gate/diagnostic。

### 7.1 第一组：复现基线

| 实验 | 目的 | 主要变量 | 结果门槛 |
| --- | --- | --- | --- |
| GigaNav policy-only | 固定 Wan/action head 的可训练性基线 | R2R、H=8 消融、无视频 loss | 复现 action loss 曲线和 1/2/3/4/5k checkpoint |
| GigaNav cotrain | 复现 video/action 共享计算 | AC-WM/WAM 混合、同样 EBS | action loss 与 video loss 分别报告 |
| V1 Stage1 visual-only | 新 token/mask/Register 的生成基线 | official Wan init、T4、IW 1/4/8/16 | 对齐旧 IWAligned visual loss 与视频质量 |
| InfiniteWorld official | 外部生成基线 | 同 prompt、同 seed、同 rollout 长度 | VBench/iWorldBench 标准输入和完整结果 |

### 7.2 第二组：结构 gate

1. `legacy_extract_update` 与 `fixed_rnull_unified RegisterCell` 对照，检查首 chunk 和长历史更新是否一致。
2. `A_cur` 正确 condition、全零 condition、去掉 condition 三组对照，检查 action-conditioned video consequence。
3. video/policy mask 对照：正确隔离、允许 A_noise 读 future、允许 A_cur 泄漏给 policy，确认泄漏设置是否只带来虚高训练指标而没有闭环收益。
4. IW-1/4/8/16 history 与 shuffled-history、zero-Register、latest-local-only 对照，检查 Register 是否真实使用历史。
5. autoregressive rollout：teacher forcing、短 autoreg、逐步增加历史长度；Register 更新保存状态，生成 chunk 只在规定的 update point 写入，避免把生成结果和 teacher history 混淆。

### 7.3 第三组：Stage2 probe 与 Stage3 replay

1. 从 Stage1 checkpoint 读取 layer 25%、50%、75%、final、Register pooled、current hidden grid，先训练轻量 pose probe，再选择正式读出层。
2. 在同一 Stage2 数据上比较 `visual-only`、`visual + pose`、不同 `lambda_pose` 和不同 probe head；必须同时记录 pose error、visual loss、decoded video 和梯度覆盖。
3. 从最佳 Stage2 checkpoint 训练 Stage3：policy-only、policy + visual replay、policy + visual + pose replay，保持完整 prefix/Register。
4. R2R train、val_seen、val_unseen、RxR 分开报告；同时做 open-loop label fit 和 closed-loop SR/SPL，避免只看 action CE。

### 7.4 多卡调度建议

| 资源组 | 任务 |
| --- | --- |
| GPU 0–1 | T4 latent/text cache 续写，优先补齐当前已确认 manifest 中未编码的 episode |
| GPU 2–3 | V1 Stage1 visual baseline 与 history bucket 对照 |
| GPU 4–5 | Stage2 layer probe sweep，冻结 backbone 先筛层 |
| GPU 6–7 | GigaNav policy/cotrain 或 Stage3 replay 对照 |

当 cache 达到可用门槛后，把 GPU0–1 转入 Stage1/Stage2 full model；不要让数据编码和训练竞争同一张卡上的显存。每个实验保留独立 `config/`、`log/`、TensorBoard、checkpoint 和 `result/` 汇总，避免多个结构写入同一曲线。

## 8. 新数据与数据组织计划

### 8.1 当前数据角色

| 数据 | 来源 | 进入哪一阶段 | action/text/3D 语义 |
| --- | --- | --- | --- |
| RE10K | 官方视频与 camera pose | Stage1/2 | pose-derived `A_hist/A_cur`；有 pose 监督；无导航 instruction 时用 empty text |
| DL3DV-10K | 官方 `images_8`、`transforms.json` | Stage1/2 | 对齐 pose 后生成 camera motion；text 通常为空 |
| SpatialVID | 官方 MP4、`poses.npy`、`indexes.txt` | Stage1/2 | 稀疏 pose 插值到 dense action；当前采用 deterministic 20% 候选 |
| Argoverse2 Sensor | 官方 front camera 与 ego trajectory | Stage1/2 | timestamp 对齐的自驾相机 motion；用于跨场景补充 |
| R2R | Matterport3D/Habitat-Sim + R2R instruction/path | Stage3 | expert action label、真实 instruction、simulator RGB |
| RxR | RxR multilingual route/instruction | Stage3 | multilingual instruction 和 expert action；当前只完成部分 budget |
| Kinetics/Sekai/Ego4D | 外部视频或第一人称数据 | 统计/未来扩展 | 当前不进入正式主线，需先完成可用视频、pose 和许可核验 |

### 8.2 数据组织原则

原始数据保持在 `/sharedata/datasets`，训练 cache 写入 `NAV/data/train/<dataset>/`。每条视频 episode 经过 source manifest、action sidecar、T4 chunk、VAE latent、text embedding 五步；Register/history 永远在 loader/forward 中在线构造。一个 episode 的 T4 latent 可以复用到 IW1/4/8/16、多种窗口位置和 Stage1/2/3 replay，不重复编码。

目前已核验的 cache 是 SpatialVID-20 视频 latent 约 185.33 GiB、R2R train latent 65.35 GiB、R2R text embedding 169.19 GiB、RxR partial latent 4.87 GiB；完整 manifest 不能直接等同于完整 latent，详细路径以 `doc/02_data/data_preparation_schema_and_status.md` 为准。

### 8.3 新数据准备顺序

1. 先把 RE10K、DL3DV、SpatialVID-20、Argoverse2 的当前候选集做成可读、可校验的 T4 latent，并生成 action statistics。
2. 继续保留 R2R full train 作为 Stage3 主数据；恢复 RxR 时优先完成 train split 和 instruction embedding，再扩展 validation。
3. 如果需要扩大动作和场景覆盖，再增加 SpatialVID 比例或加入 Argoverse2；扩容前先保持 dataset-aware sampler，避免 SpatialVID 的 episode/chunk 数压倒其他数据。
4. Sekai、Ego4D 等待凭据后单独建立新 manifest，完成小规模完整链路核验，再决定是否进入训练；不能只依据下载文件大小纳入主线。

## 9. 新训练计划与验收顺序

### Phase 0：数据和链路冻结

- 固定数据 manifest、T4 latent shape、text fallback、action combo schema、STOP padding 和 history sampler。
- 对每个数据集跑完整 loader batch audit：检查 `C_hist/Z_obs/Z_future/A_hist/A_cur/A_noise/text/pose/action_label` 的形状、时间索引和 mask。
- 用随机初始化和 Wan official init 分别跑完整 forward/backward smoke；只确认链路，不把 smoke loss 当效果结论。

### Phase 1：V1 Stage1 生成记忆

- 从 official Wan2.1-1.3B 初始化，训练 shared WanBlock、Register 和 video branch。
- 混合 IW1/4/8/16 history，随机窗口和 dataset-aware sampling；先 teacher-forced，再逐步增加 autoreg rollout。
- 每 1,000 steps 保存 checkpoint；在 1k/5k/10k 或验证集平台期做相同 prompt、相同 seed 的 decode 和 VBench/iWorldBench。
- 只有 visual loss、视频质量和长历史 memory ablation 同时通过，才进入 Stage2。

### Phase 2：V1 Stage2 空间监督

- 先 probe sweep 冻结 backbone，选择 current hidden/Register 的最佳层。
- 再从 Phase1 最佳 checkpoint 进行 visual + pose 训练，保留 zero-action condition 比例。
- 检查 pose error 是否低于 no-3D、visual 是否不退化、Register 是否在 IW16 仍有历史增益。

### Phase 3：V1 Stage3 导航后训练

- 使用 R2R full train，逐步加入 RxR；完整 prefix 在线更新 Register，policy 每步只读 Register、当前 observation、instruction 和 action slots。
- 先做 action-only fit 作为数据/label gate，再加入 visual/pose replay；action CE 需要与 majority baseline、open-loop 和 closed-loop 指标一起看。
- 最终报告 R2R val_seen/val_unseen、RxR 对应指标、long-history bucket、policy-only latency、video/pose replay 保持率。

### Phase 4：论文主实验与 ablation

至少完成以下对照后再冻结论文结论：

```text
full NAV
  vs local-only
  vs zero/shuffled Register
  vs HPMC / InfiniteWorld baseline
  vs no 3D supervision
  vs no A_cur consequence condition
  vs no replay
  vs historical dual-stream/GigaNav ablation
```

每项结果都需要绑定代码版本、配置、checkpoint、数据 manifest、TensorBoard 和结果路径；概念性结论按项目规则写成 `【已验证→...】` 或 `【未验证】`。

## 10. 当前最合理的执行优先级

第一优先级是把 V1 Stage1 的完整生成支路、T4 数据和 history sampler 固定下来，并以旧 IWAligned/InfiniteWorld 做同协议回归。第二优先级是用冻结主干 probe 找到能读出 pose 的 hidden 层，再做 Stage2 mixed loss。第三优先级是用 R2R full train 验证 action label 和 Register prefix，最后加入 replay 并扩展 RxR。

如果多卡资源足够，建议同时运行一条 GigaNav action baseline、一条 V1 Stage1 generation baseline 和一条 Stage2 probe sweep；三者的结果可以把“backbone/action 能否训练”“生成支路是否稳定”“3D 表征是否可读”三个问题分开，避免再次把单一 loss 曲线当成整个方案有效性的证据。
