# NAV 文档维护约定

本文件约束 `NAV/doc/` 下的新增、迁移和更新。

## 必须遵循

1. 主体说明使用中文；模型、数据集、指标和模块名保留英文。
2. 文件名使用 `lower_snake_case.md`，目录使用两位编号加语义名。
3. 每篇正式文档必须有唯一 `NAV-<TYPE>-NNN` ID、类型、状态、更新时间和职责。
4. 每篇文档只能有一个一级标题；正文从二级标题开始。
5. 先写结论、当前状态或执行入口，再写背景和推导。
6. 数量、耗时、显存、磁盘和指标必须注明统计口径。

## 唯一事实来源

- 当前状态总览：`00_overview/project_status.md`
- 当前 V1 硬规则：`00_overview/project_invariants.md`
- 当前 V1 模型接口：`01_design/v1_action_centered_io_interface.md`
- 当前 V1 待定项：`01_design/v1_open_design_questions.md`
- 下载和预处理状态：`03_data/dataset_preparation_status.md`
- V1 数据与训练变量表：`04_training/v1_three_stage_training_data_plan.md`
- V1 Stage One T4/IW 对齐训练：`04_training/v1_stage_one_t4_iw_aligned.md`
- 资源路径：`06_operations/resource_inventory.md`
- 关键方案变更：`00_overview/decision_log.md`

## V0/V1 阅读规则

- `v1_*` 是当前主线：Wan2.1 official init、`T_latent=4`、Register /
  action-centered interface、Stage One/Two/Three 新三阶段。
- `v0_*` 是历史与对照：InfiniteWorld checkpoint init、81-frame dense chunk、
  A/B Register 旧实验、VBench baseline 和旧 Stage Two/Three 设计。
- 无 `v0/v1` 前缀文档必须在正文说明它是通用规范、资源状态，还是历史兼容入口。
- 写新实验记录时，如果实验属于当前正式主线，优先新增或更新 `v1_*` 文档；
  只有复现旧结果、对照或解释历史 run 时才写入 `v0_*`。

## V1 数据准备显存规则

数据准备类 GPU 工作默认遵循“尽量吃满显存但不允许持续 OOM”的规则：

1. 使用全局 shard index 分片，多个 worker 不得重复处理同一批样本。
2. 每个 worker 写独立输出 root，再由 multi-root dataset reader 汇总。
3. RTX 6000 Ada 当前 V1 VAE pack 编码校准为 GPU0 7 workers、GPU1 7 workers；
   GPU0 7 + GPU1 8 曾触发 CUDA OOM，因此 7x7 是当前“吃满但稳定”的默认点。
   若出现 OOM，优先降低 worker 数而不是让错误样本继续累计。
4. 检查命令以 `scripts/check_v1_vae_pack_encode.sh` 或当前 T4 micro latent
   专用检查脚本为准。

`repointro.md` 只是兼容入口，不得重复维护资源内容。

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
   记录）后才能进入 essay。未收敛的设想写在 `doc/08_insight/`，不进 essay。
3. **不一致即停**：若 essay 与 doc/代码/实验结果不一致，以 doc 与实验为准，
   先修正 essay，不得反向用 essay 覆盖项目事实。
4. **引用回溯**：essay 中引用的数字与图表应标注其 `doc/` 或 `result/`
   来源路径，便于核对。

## 修改后检查

1. 新文档是否加入 `README.md` 注册表。
2. 路径是否同步到资源总账。
3. 方案变化是否新增决策日志条目。
4. 训练或评测口径是否同步到 config、script 与结果说明。
5. 是否仍有旧路径、多个一级标题或失效的本地引用。

## 概念性结果标注规则

essay（`intro`、`relatedworks`、`experiment`）与 doc 中所有"概念性结果"
（即对方法/能力/对比给出的结论性陈述，如"双重 probe 证明几何可读""Register
优于 HPMC""拿掉 3D 监督后退化"等）必须就地标注验证状态：

- **已验证**：`【已验证→<记录指针>】`，指针指向 `result/` 产物路径、
  `doc/05_evaluation/` 记录号或 `log/` checkpoint，且该产物真实存在。
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
3. 分辨率、batch size、worker 数：在不改变代码路径语义的前提下降低资源需求。

不允许替代：

1. 不允许用 `scaffold` 模型、临时小 Transformer、旁路小 head 或 mock backbone
   证明正式模型已跑通。
2. 不允许绕过正式 dataloader、正式 model forward、正式 loss、正式 inference /
   eval entrypoint 后，把结果记为正式 smoke。
3. 不允许把历史 scaffold run 当作当前 V1 正式结构的速度、显存、loss 或质量结论。

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

`08_insight/insight_log.md` 记录对 NAV idea 与网络设计的改进分析，受以下
规则约束：

1. 每次新分析前必须先把此前所有问题压缩进该文件的"问题寄存器"表并标注
   状态（`Answered` / `Open` / `Superseded`）。
2. 只有当问题寄存器中所有 `Open` 问题都变为 `Answered`（含实验证明、决策
   采纳或显式废弃）后，才允许开启下一轮新分析。
3. Insight 主体必须详细：动机、对应已调研工作、具体改法（不动 Wan2.1
   backbone）、预期收益、可验证实验指针、风险。
4. Insight 不等于决策；被采纳并写入 `01_design/` 或 `04_training/` 且在
   `decision_log.md` 记录后才成为正式方案。
