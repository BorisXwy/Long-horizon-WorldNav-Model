# NAV Agent 工作约定

本文件约束 agent 在 `NAV/` 项目中的代码、文档、实验和写作工作；其中 `doc/` 是知识库，`log/` 与 `result/` 是运行产物目录。

## 必须遵循

1. 主体说明使用中文；模型、数据集、指标和模块名保留英文。
2. 文件名使用 `lower_snake_case.md`，目录使用两位编号加语义名。
3. 每篇正式文档必须有唯一 `NAV-<TYPE>-NNN` ID、类型、状态、更新时间和职责。
4. 每篇文档只能有一个一级标题；正文从二级标题开始。
5. 先写结论、当前状态或执行入口，再写背景和推导。
6. 数量、耗时、显存、磁盘和指标必须注明统计口径。

## Git 提交与推送规则

自 2026-08-21 起，agent 每次修改 NAV 项目的代码或文档后，必须在本轮工作
结束前完成对应的 Git commit，并推送到远端仓库。commit message 必须明确记录
本次代码/文档改动与实验目的，例如涉及 Stage2 probe 时需写清：

```text
stage2 probe sweep: add frozen-backbone layer probes and docs
```

执行要求：

1. 只提交本次任务相关文件，避免把历史未整理改动、他人改动或大文件产物混入。
2. commit 前使用 `git status --short` 检查 staged 范围。
3. 训练日志、checkpoint、视频、latent、rendered PNG 等大产物不得提交；只提交
   可复现脚本、配置、文档和小型文本摘要。
4. 若远端 push 失败，必须在最终回复中说明失败原因、已生成的本地 commit hash
   和需要用户处理的事项。
5. 若工作树已存在大量无关 dirty 文件，必须显式采用 pathspec 精确 add 本次文件。

## 唯一事实来源

- 当前状态总览：`doc/00_overview/project_status.md`
- 项目规则与文档规范：`doc/00_overview/project_rules_and_documentation_standard.md`
- 最终正式结构、训练与 acceptance 硬标准：
  `doc/00_overview/final_formal_training_standard.md`
- 当前 V1 模型结构、接口、历史模型尝试与待定项：
  `doc/01_model/model_evolution_and_current_architecture.md`
- 下载、预处理、Action/Geometry Schema 与 VLN 渲染状态：
  `doc/02_data/data_preparation_schema_and_status.md`
- V1/V0 训练计划、Stage One/Two/Three、课程与实验记录：
  `doc/03_training/training_plan_and_experiment_log.md`
- 评测协议、外部复现与 benchmark 调研：
  `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`
- Insight 问题寄存器：`doc/05_insight/insight_log.md`
- 资源路径：`doc/06_operations/resource_inventory.md`
- 关键方案变更：`doc/00_overview/decision_log.md`

## V0/V1 阅读规则

- V1/V0 不再按大量分散文件维护，而是进入主题总文档中的对应章节。
- `V1` 是当前主线：Wan2.1 official init、`T_latent=4`、Register /
  action-centered interface、single shared WanBlock、Stage One/Two/Three 新三阶段。
- `V0` 是历史与对照：InfiniteWorld checkpoint init、81-frame dense chunk、
  A/B Register 旧实验、VBench baseline 和旧 Stage Two/Three 设计。
- 新实验记录优先写入对应主题总文档；只有内容很大且会长期独立维护时才新增文件。

## V1 数据准备显存规则

数据准备类 GPU 工作默认遵循“尽量吃满显存但不允许持续 OOM”的规则：

1. 使用全局 shard index 分片，多个 worker 不得重复处理同一批样本。
2. 每个 worker 写独立输出 root，再由 multi-root dataset reader 汇总。
3. RTX 6000 Ada 当前 V1 VAE pack 编码校准为 GPU0 7 workers、GPU1 7 workers；
   GPU0 7 + GPU1 8 曾触发 CUDA OOM，因此 7x7 是当前“吃满但稳定”的默认点。
   若出现 OOM，优先降低 worker 数而不是让错误样本继续累计。
4. 检查命令以 `scripts/check_v1_vae_pack_encode.sh` 或当前 T4 micro latent
   专用检查脚本为准。
5. Stage3 VLN 数据准备遇到 terminal `STOP` 时必须按吸收态处理：`STOP` 后的
   action target 填充为 `STOP STOP ...`，后续 RGB frame / latent frame 复制
   terminal observation。当前渲染默认 `post_stop_min_frames=97`、
   `post_stop_pad_frames=12`，用于保证短 episode 也能构造 IW1 history + current
   obs 的 policy window；预算 manifest 必须使用同一口径估算 micro chunks。
6. 自 2026-08-22 起，训练用 latent 的真实落盘位置改为个人项目目录
   `NAV/data/train/<dataset>/<latent_run>/`，例如
   `NAV/data/train/rxr_ce/t4_micro_latents_stoppad_20260822_1423/`。`/sharedata`
   只保留原始下载数据、simulator assets、预算 manifest 和必要的临时 render
   中间态；不得再把新的大规模训练 latent 默认写入 `/sharedata/NAV/derived/`。
   `NAV/data/` 已由 `.gitignore` 忽略，不得提交其中的 `.pt`、PNG 或大 manifest。

`doc/repointro.md` 只是兼容入口，不得重复维护资源内容。

## V1 架构硬规则

0. **2026-08-18 起的最终正式训练标准**：完整标准单独维护在
   `doc/00_overview/final_formal_training_standard.md`。后续所有声称为“最终结构 / 正式训练 /
   acceptance”的 run，必须同时满足以下条件；只满足其中一部分的 run 只能标为
   `gate`、`ablation` 或 `diagnostic`：
   - 使用新版结构：保留多类 token 交互，包含 video generation branch 与
     policy branch，并显式维护二者的 attention / causal mask 关系；
   - 保留 `A_noise -> A_out` policy/action path；Stage1/2 可以不监督 action，
     但 token 位置、mask、forward path 不得删除；
   - `A_hist` 用于 Register update；`A_cur` 作为 video generation 的
     action condition；`A_noise/A_out` 用于 Stage3 policy/action supervision；
   - Register update 与 action condition 训练时必须混入一定比例的 all-zero
     empty input，用于覆盖无 action / 无条件场景；
   - latent 规则为 `T_latent=4`；history 长度混合必须对齐 InfiniteWorld
     `1 / 4 / 8 / 16` IW-chunk 等价长度；
   - 数据混合至少包含 `RE10K + SpatialVID + DL3DV`；有 text 的样本直接编码
     text，没有 text 的样本使用此前约定的空白/empty text embedding；
   - Stage2/正式 co-training 使用 `3D + visual` 混合 loss；Stage3 在此基础
     上加入 policy/action loss，并保留 visual/3D replay 以防退化；
   - effective batch size 默认 `ebs=16`。如果因显存只能降级，必须显式标注
     为资源降级 run，不得与正式标准混淆。
   - 昨日 `formal_v1_fixed_rnull_shared_tail_1k_fixedpatch_20260818_013124`
     只证明 `fixed_rnull_unified + shared_tail_token` 子结构在 RE10K Stage2
     gate 上可收敛，不等同于上述最终正式训练。

1. 当前正式 V1 主结构遵循 DEC-041：**single shared WanBlock token stream**。
   主 DiT / policy-visible token stream 是 Register、`Z_obs`、future、text、
   `A_cur/A_noise`。`A_hist` 只用于 Register update；Register 已吸收历史动作后，
   policy 侧不得再直接读取独立 `A_hist` 或原始 `Z_history` token。
2. 不得把 GigaWorld-style dual-stream / MoT-style action expert 当作正式主线
   实现；它只能作为 related work、历史 diagnostic 或显式 ablation。
3. 保留最新 action interface：所有样本只暴露一种 action；正式 schema 是
   `combo_id = trans_id * 12 + rot_id`，policy head 输出
   `combo_logits=[B,H_action,144]`。Stage One/Two 视频数据把 action 用作
   `a_condition / A_hist / A_cur`，Stage Three VLN 数据把 action 用作
   `a_label / A_out` 监督。不得新增独立 `trans_logits` 与 `rot_logits` 作为
   默认正式监督头。
4. `move/view` 是 pose-derived camera-motion factorization，不是所有数据集
   共享的 universal robot action ontology。涉及 lateral / strafe / “蟹行”
   时必须先查 `result/data_audit/crabwalk_motion_20260817.json` 或重新运行
   `scripts/analyze_crabwalk_motion.py`；R2R/VLN 默认不含 strafe primitive。
5. 若代码、日志或旧 smoke 中仍出现 `DualStreamBackbone` / 独立 action expert，
   必须标为 `Superseded by DEC-041`，不得继续开正式训练。

## V1 源代码模块化规则

1. 当前 canonical 模型定义位于 `src/nav/v1/model/`：组件类分别放入
   `world_model.py`、`policy.py` 等模块，最终构图统一由 `builder.py` 的
   `build_model()` 完成。训练和推理脚本不得重复 checkpoint 恢复或模型组装逻辑。
2. 当前 canonical 数据定义位于 `src/nav/v1/data/`：dataset/window 逻辑放在
   `stage2.py`、`r2r.py` 等模块，采样策略放在 `sampler.py`，最终由
   `loader.py` 的 `build_data_loader()` 组装。脚本不得内联复制 sampler/loader。
3. 新 ablation 应新增或注册独立 assembler、module、loader 或 sampler，再通过
   名称组合；不得在一个训练脚本中持续叠加条件分支，导致正式构图随脚本漂移。
4. `src/nav/v1/stage2_final/`、`stage3_r2r.py`、`stage3_single_action.py` 是旧路径
   compatibility shim，只允许重导出 canonical 类，不得再加入新实现。
5. 纯目录重构不得改变默认 config、数据 window/sampling、张量形状、forward、loss、
   state_dict key 或 checkpoint payload。重构验收至少包含真实 loader 逐张量对照、
   正式 checkpoint 严格兼容加载和完整 Wan 前向数值对照。
6. GigaWorld-style navigation ablation 必须与 V1 主线隔离：核心模型和数据组装
   放在 `src/giga_nav/`，可执行训练/评测入口放在 `scripts/`。该 ablation 使用本地
   Wan2.1-T2V-1.3B（不得误加载 Wan2.2-5B），保留 Giga 的
   `[state, reference visual, action, noisy future]` 共享 token 计算；仅因 R2R
   标签改用四类离散 CE，且缺失 proprioceptive state 时显式输入零 state token。
   不得用 toy/scaffold backbone 代替完整 Wan 前向；规则、张量接口和实测结果同步
   维护于 `doc/01_model/giga_nav_ablation_wan21_13b.md`。

## 内容边界

- 原始运行日志写入 `NAV/log/`。
- 指标、视频和评测产物写入 `NAV/result/`。
- 论文写作（核心 `.tex` 主文件、章节子文件、图表源码、草稿）写入
  `NAV/essay/`；不在此处存放原始日志、checkpoint 或评测 JSON。
- 文档只保留可复现摘要：命令、配置、数据版本、checkpoint、结果和异常结论。
- 不记录 token、cookie、AWS key、订阅链接或其他私密凭据。
- 历史内容不得无说明删除；使用 `Superseded` 和决策日志记录替代关系。

## Essay 内容来源规则

`NAV/essay/` 的论文内容必须来自已收敛的有效项目内容，不得先于项目验证
而凭空撰写：

1. **来源链**：essay 中的主张、方法描述、数字与图表，必须可回溯到
   `doc/`（设计/规范/决策）与 `result/`（实验产物），并经核心代码
   （`src/`）与实验运行（`log/`）实际验证。
2. **收敛前置**：任何写入 essay 的方法、结论或实验结果，必须先在
   `doc/` 与核心代码/实验中收敛（设计已定、代码已实现、实验已跑出并
   记录）后才能进入 essay。未收敛的设想写在 `doc/05_insight/`，不进 essay。
3. **不一致即停**：若 essay 与 doc/代码/实验结果不一致，以 doc 与实验为准，
   先修正 essay，不得反向用 essay 覆盖项目事实。
4. **引用回溯**：essay 中引用的数字与图表应标注其 `doc/` 或 `result/`
   来源路径，便于核对。

## 修改后检查

1. 新文档是否加入根目录 `README.md` 的文档注册表。
2. 路径是否同步到资源总账。
3. 方案变化是否新增决策日志条目。
4. 训练或评测口径是否同步到 config、script 与结果说明。
5. 是否仍有旧路径、多个一级标题或失效的本地引用。

## 概念性结果标注规则

essay（`intro`、`relatedworks`、`experiment`）与 doc 中所有"概念性结果"
（即对方法/能力/对比给出的结论性陈述，如"双重 probe 证明几何可读""Register
优于 HPMC""拿掉 3D 监督后退化"等）必须就地标注验证状态：

- **已验证**：`【已验证→<记录指针>】`，指针指向 `result/` 产物路径、
  `doc/04_evaluation/` 记录号或 `log/` checkpoint，且该产物真实存在。
- **未验证**：`【未验证】`，并尽量补一句未验证的原因（如"Stage Two 未实现"
  "仅短程 proxy""长程 autoreg 未跑"）。背景性陈述（领域现状、他人工作描述）
  不算概念性结果，无需标注。

标注必须与 `result/`/`log/` 实际产物一致；无产物支撑的结论一律标 `【未验证】`，
不得以"实验设计"代替"实验产物"作为已验证依据。此规则与下文 Essay 内容来源
规则联动：未验证结论不得在 essay 中以已证口吻叙述。

## Agent 工作三结合规则

agent（含自动化训练、评测、调研、写作）的任何工作必须**结合代码、记录、
文档三线**，不得只凭其一：

1. **代码**（`src/`、`scripts/`、`config/`）：主张的能力必须有对应实现；
   引用模块/脚本时确认其真实存在。
2. **记录**（`log/` 训练日志与 checkpoint、`result/` 评测产物）：任何"已
   验证"结论必须有真实产物指针，且数字/路径与产物一致。
3. **文档**（`doc/`）：代码与记录的变更必须同步落 doc；doc 的主张必须可
   回溯到代码+记录。

三线不一致时以"代码+记录"为准先修 doc，不得反向用 doc/essay 覆盖工程事实。
新增/修改能力时，agent 须同步检查三线是否收口，并在 doc 标注验证状态。

## 完整链路 smoke 规则

之后所有训练、测试、推理的 smoke test / 冒烟测试必须使用正式代码和完整流程，
不得再使用 scaffold / toy path / mock module 作为有效验证依据。

允许缩小：

1. 数据量：例如只取 1-2 个 batch、少量 episode、少量 prompt。
2. 步数：例如 1 step / 10 steps / 小规模 eval subset。
3. batch size、gradient accumulation、worker 数：在不改变模型结构、输入输出
   语义和数据变量的前提下降低资源需求。

不允许替代：

1. 不允许用 `scaffold` 模型、临时小 Transformer、旁路小 head 或 mock backbone
   证明正式模型已跑通。
2. 不允许绕过正式 dataloader、正式 model forward、正式 loss、正式 inference /
   eval entrypoint 后，把结果记为正式 smoke。
3. 不允许把历史 scaffold run 当作当前 V1 正式结构的速度、显存、loss 或质量结论。
4. 不允许为了冒烟或快速验证而缩小模型结构或关键张量几何，包括但不限于：
   backbone depth、hidden width、attention/token layout、Register token 数、action
   horizon、`T_latent`、latent/video 空间分辨率、VAE 编解码分辨率、loss 分支变量。
   如因调试临时缩小，必须标为 `diagnostic invalid for acceptance`，不得作为
   “完整训练”“正式 smoke”“验收结果”或方案有效性证据。

记录 smoke 时必须写清：

```text
正式入口脚本 / config
真实加载的模型权重或初始化方式
真实数据 root / manifest
是否执行完整 forward / backward / loss / decode / metric
缩小了哪些规模参数
输出 log / result 路径
```

若因为工程尚未完成只能跑 scaffold，必须明确标为 `scaffold diagnostic` 或
`toy diagnostic`，不得标为 `smoke verified`、`implemented` 或正式训练/测试依据。

## Insight 分析规则

`doc/05_insight/insight_log.md` 记录对 NAV idea 与网络设计的改进分析，受以下
规则约束：

1. 每次新分析前必须先把此前所有问题压缩进该文件的"问题寄存器"表并标注
   状态（`Answered` / `Open` / `Superseded`）。
2. 只有当问题寄存器中所有 `Open` 问题都变为 `Answered`（含实验证明、决策
   采纳或显式废弃）后，才允许开启下一轮新分析。
3. Insight 主体必须详细：动机、对应已调研工作、具体改法（不动 Wan2.1
   backbone）、预期收益、可验证实验指针、风险。
4. Insight 不等于决策；被采纳并写入 `doc/01_model/` 或 `doc/03_training/` 且在
   `doc/00_overview/decision_log.md` 记录后才成为正式方案。
