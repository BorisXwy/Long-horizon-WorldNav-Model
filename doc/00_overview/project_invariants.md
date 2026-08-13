# NAV 项目硬规则与不变量

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-004` |
| 类型 | 规则声明（Project Invariants） |
| 状态 | Active / Guardrail |
| 更新时间 | 2026-08-14 |
| 职责 | 汇总当前 V1 主线中不应被普通实验随意改变的模型、数据、训练、评测和文档硬约束 |

## 当前结论

本文不是实验记录，也不是完整设计推导；它只保存“改代码、开训练、写论文时不能
悄悄违反”的硬规则。若确实要改变其中任何规则，必须先在
`decision_log.md` 新增决策，并同步修改对应设计、训练、数据和脚本文档。

## 版本边界

1. `v1_*` 是当前主线：official Wan2.1-1.3B initialization、`T_latent=4`
   micro future、Register long-history memory、action-centered interface、
   Stage One/Two/Three staged training。
2. `v0_*` 是历史与 baseline：InfiniteWorld checkpoint initialization、
   81-frame dense chunk、HPMC、早期 A/B Register 和 VBench 复现。
3. V1 正式训练不得在没有说明的情况下加载旧 RE10K A/B、旧 DL3DV A/B 或
   InfiniteWorld finetuned checkpoint。
4. InfiniteWorld checkpoint 只用于 V0 baseline、旧实验复现或显式 ablation；
   V1 正式 Stage One 默认从官方 Wan2.1 权重初始化。

## 模型不变量

1. V1 shared backbone 是 Wan2.1-style DiT；Register、generation branch 和
   policy branch 共享 backbone，而不是给 policy 另建一个完全独立视觉 backbone。
   当前正式结构采用 dual-stream / MoT-style backbone：visual stream 继承 Wan
   video token 计算，action stream 使用独立 action expert / decoder，并在受控
   attention 关系下交互。
2. HPMC history compression 在 V1 主线中被移除或旁路；long history 由
   Register 递归承载。
3. InfiniteWorld 的 `20-channel mask` 在 V1 正式结构中删除；DiT patch stem
   回到 Wan 原生 `16 latent channels -> hidden dim`。prefix / future / action
   的身份由 stream/type embedding、token_type_embedding 和 attention split /
   mask 显式表达，不再用 channel concat 表示。
4. latest local memory 保留；它表示最近局部视觉状态，不承担无限历史压缩职责。
5. Register 不保存 episode-specific state 到 checkpoint；checkpoint 保存的是
   RegisterCell、DiT、action interface / heads 等参数。
6. 新 episode 的初始状态固定为 `R_null`，不是 learnable episode memory；
   `R_null` 只提供固定 register slots / type-position scaffold。第一个 history
   chunk 也通过同一个 RegisterCell 写入：
   `R_0 = RegisterCell(R_null, concat([visual_tokens(C_0), A_hist_0]))`；后续递归为
   `R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`。
7. 训练和推理中不得把 generated future 作为导航 history 回灌；导航 history
   只来自真实 observation。

## Action 与 Pose 不变量

1. 三类 action 语义必须分开：
   - `A_hist`：历史已执行 action / pseudo motion，参与 RegisterCell 更新，
     但必须以独立 action tokens / cross-attention context 的形式交互；
     禁止 action bias / latent bias 注入；
   - `A_cur`：当前 video generation condition，用于约束 future visual consequence；
   - `A_noise -> A_out`：policy/action output 路径，采用 Giga/DreamZero-style
     action flow / diffusion decode；Stage Three 才用 VLN action label 监督。
2. Stage One 不把 video pseudo action 训练成 policy；`A_out` 没有 action
   supervision。`A_cur` 必须作为 DiT condition/context 进入 video generation
   branch，并且只作用于 noisy future visual tokens，不作用于 history prefix 或
   `A_noise`。`A_noise` 才作为 action stream 主变量进入 Wan/DiT backbone。
   禁止再使用“Register/local -> 旁路小 action head”冒充
   policy/video shared backbone。
3. Register 由统一的 `RegisterCell` 递归更新，不再区分 Extractor / Updater：
   `R_{-1}=R_null`，
   `R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`。
   `A_hist` 只能通过 token concat、cross-attention 或 gated token adapter 参与
   Register update；不得把 action
   embedding 加到 latent、Register、patch tokens 或 DiT hidden 上作为 additive
   bias，也不得使用 InfiniteWorld-style action bias / latent bias 方案。Pose、
   odometry 和 pseudo motion 只能用于构造 `A_hist` 标签/embedding 或训练监督，
   不作为额外 Register 输入通道。
4. `A_noise` 属于 action stream 中的 noised action tokens；future visual tokens
   可以 attend 到它，action tokens 不得 attend 到 future visual tokens。
5. timestep 采用 GigaWorld-Policy-style per-token 语义：
   - clean prefix / Register / local visual tokens：`t=0`；
   - future noisy visual tokens：`visual_t`，由 RFlowScheduler 采样并用于
     Stage One video flow loss；
   - `A_noise` action tokens：独立 `action_t`，默认
     `action_flow_shift=5.0`；
   - Stage One 中 `A_out` 不计算 action loss，`action_t` 只用于让 action token
     格式与 Stage Three policy/action denoise 路径对齐。
6. 默认 action horizon 固定为 `H_action = H_nav = 10`。闭环导航推理可以只执行
   denoise 后 action chunk 的第一个 action，但训练和接口统一保留 10-step
   action output。
7. Pose 不是模型推理输入。Pose 只能用于 pseudo action、3D supervision、
   数据审计和尺度校准。
8. Stage Three policy 输入不得依赖 GT pose；真实世界没有 GT pose 是默认假设。
9. 视频 pseudo action 与 VLN discrete action 的尺度和语义必须显式映射，不得
   把二者当作天然同分布标签。

## 数据与缓存不变量

1. V1 Stage One 当前正式缓存是 T4 micro latent，不是 V0 的 81-frame dense
   latent，也不是早期 sparse pack scaffold。
2. 每个 T4 micro chunk 固定：
   - RGB frames：13；
   - stride：12；
   - VAE latent time：`T_latent=4`。
3. 每个 micro chunk 必须独立调用 Wan VAE encoder。不得先整段视频 VAE 编码
   再切 latent。
4. T4 micro latent 可复用于不同 history window。样本构造时从已缓存 episode
   latent 中抽取窗口，不重复编码。
5. 对齐 InfiniteWorld 发生在 history/context span，而不是 target 长度：

   | IW-equivalent history | T4 micro history steps |
   | --- | ---: |
   | IW-1 | 7 |
   | IW-4 | 27 |
   | IW-8 | 54 |
   | IW-16 | 107 |

6. SpatialVID 数量天然占优，正式训练不得无说明地按自然文件顺序或自然样本量
   直接吞；必须记录 sampler 或数据比例。
7. 原始下载目录保持只读；派生数据写入 `/sharedata/NAV/derived/`。

## 三阶段训练不变量

1. Stage One 是 video generation memory pretraining：
   `L_stage1 = L_visual_flow`，训练 Register long-history update 和 short
   future generation。
2. Stage Two 不是 cotrain/replay 命名；它是在 Stage One 同一视频生成前向中
   加 3D supervision/probe：
   `L_stage2 = L_stage1 + lambda_3d * L_3D`。
3. Stage Three 是 VLN policy/action supervision，并需要 replay Stage One/Two
   loss 防止 shared backbone/Register 退化：
   `L_stage3 = L_nav + lambda_video * L_visual_flow_replay + lambda_3d * L_3D_replay`。
4. Stage Three 推理是 policy-only：generation/video/3D heads 不执行。
5. 训练启动时扫描到的数据列表通常是固定的；后台新落盘 latent 不自动进入已启动
   训练，除非脚本显式支持动态 refresh，并在日志中记录。

## 评测与结论不变量

1. 不能仅凭 training loss 宣称 long-history memory 有效。
2. 声称 Register 长程有效前，至少需要：
   - normal Register vs zero Register；
   - true history vs shuffled history；
   - with local memory vs without local memory；
   - IW-1/4/8/16 分组指标；
   - teacher-forced 与 autoreg rollout 对比。
3. 声称优于 InfiniteWorld 前，必须在同一数据、同一 history span、同一生成/评测
   协议下比较。
4. 文档和 essay 中所有概念性结论必须标注 `【已验证→路径】` 或 `【未验证】`。

## 文档和运行不变量

1. 当前事实先看 `project_status.md`；路径先看 `resource_inventory.md`；原因先看
   `decision_log.md`。
2. 代码、记录、文档三线不一致时，以代码和真实 log/result 为准，先修文档，再
   继续正式实验。
3. 新训练必须记录：
   - checkpoint initialization；
   - loaded/missing/mismatched key audit；
   - data root 和 manifest；
   - history construction；
   - loss 组成；
   - batch、accumulation、optimizer、dtype；
   - save frequency；
   - TensorBoard/log/checkpoint 路径。
4. 结果类内容放在 `NAV/result/<eval_type>/<run_name>/...`；训练原始日志和
   checkpoint 放在 `NAV/log/`；不要把产物塞进 `doc/`。
