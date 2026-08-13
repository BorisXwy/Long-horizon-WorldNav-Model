# R2R / RxR Benchmark 与最新 SOTA 调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-002` |
| 类型 | Benchmark / Leaderboard 调研 |
| 状态 | Verified Snapshot / External Results |
| 更新时间 | 2026-08-07 |
| 职责 | 统一 R2R、R2R-CE、RxR、RxR-CE 的评测口径，核对论文引用方式，并给出 NAV 使用的最终成功率基准 |

## 本地复现队列（2026-08-07）

NAV 实际 Habitat 复现只覆盖 **R2R Val-Unseen SR ≥ StreamVLN** 且有公开权重的方法（含
StreamVLN）。工程约束与命令见 **`../05_evaluation/r2r_ce_sota_reproduction.md`
（NAV-EVL-002，DEC-021）**：仓库在 `3d_wm_vln/{StreamVLN,InternNav}`，venv 在
`virtual_env/.venv_{streamvln,internnav}`。Qwen-RobotNav / Robostral 等更高分但无权重
的工作仍只作本文件的引用追踪，不进入复现队列。

## 当前结论

1. NAV 当前本地数据和 Habitat 导航语境对应的是 **R2R-CE / RxR-CE**，不能把
   离散 navigation graph 上的 R2R / RxR 数字直接作为基准。离散 R2R 的 Test SR
   已达到约80%，明显高于连续环境，不代表低层控制成功率。
2. 论文中的 SOTA 表格大量直接引用前人论文或 leaderboard 数字，特别是隐藏 GT 的
   Test split；只有标注 `*`、`reproduced`、`our implementation` 的行才是作者重跑。
3. Test GT 不公开，因此论文作者无法在本地自行计算官方 Test SR。自己的 Test
   数字原则上来自 leaderboard submission；前人 Test 数字通常直接复制前人论文或
   leaderboard。Val-Unseen 可本地评测，但很多论文仍直接引用前人数字。
4. 官方 VLN-CE 仓库明确记录过：同一 CMA baseline 在原论文中报告
   Val-Unseen SPL=0.30，而 leaderboard 重评为0.27，并建议今后以 leaderboard 为准。
   因此“表格中引用过”不等于“在当前 Habitat 版本上重新复现过”。
5. 截至2026-08-01，建议 NAV 对外采用以下**最终基准**，同时保留 split 与来源：

| Benchmark | 推荐主口径 | 最终 SR | 同时报告 | 证据等级 |
| --- | --- | ---: | --- | --- |
| R2R-CE | Test-Unseen，同行评审、官方 Test 评测 | **60%** | SPL 52% | NavMorph ICCV 2025，HNR backbone |
| R2R-CE | Val-Unseen，最新同行评审 claim | **64%** | SPL 59% | HSAN NeurIPS 2025；非官方 Test leaderboard |
| RxR-CE | Test-Challenge，公开挑战赛确定值 | **45.82%** | SPL 38.82%，NDTW 55.43% | Reborn，RxR-Habitat 2022 winner |
| RxR-CE | Test-Unseen，后续同行评审论文值 | **54.98%** | SPL 43.02%，NDTW 57.31%，SDTW 44.76% | NavMorph ICCV 2025，HNR backbone |

这里不能压缩成一个不带限定词的“最终成功率”。若只为 NAV 设定两个连续环境目标，
使用 **R2R-CE Test SR=60%**、**RxR-CE Test SR=54.98%**；若要求严格对应公开挑战赛
历史榜单，则 RxR-CE 应改用 **45.82%**。
6. 最新的 **Qwen-RobotNav**（arXiv:2606.18112，2026-06，Technical Report）将
   Val-Unseen 推进到 R2R-CE **72.1/66.6**、RxR-CE **76.5/65.7**（SR/SPL，
   panoramic，8B）。这是当前最值得 NAV 跟踪的统一导航基础模型，但它没有公开权重，
   且这些数值不是官方 Test submission，故只进入 Val-Unseen 追踪表。
7. 截至当前检索日，更新的单目 preprint **Robostral Navigate**（arXiv:2607.20785）
   自报 R2R-CE **77.4% SR**、RxR-CE **75.1% SR**。它在 R2R 单目口径超过
   Qwen-RobotNav，但尚未同行评审，也不能与 panoramic Qwen-RobotNav 或 Test 主表混用。

## Benchmark 边界

| 名称 | 环境与动作 | 语言 | 常用 split | 首要指标 |
| --- | --- | --- | --- | --- |
| R2R | Matterport3D navigation graph，节点间跳转 | English | Val-Unseen / Test-Unseen | SR、SPL |
| R2R-CE | Habitat 连续空间，低层离散动作 | English | Val-Unseen / Test-Unseen | SR、SPL |
| RxR | Matterport3D navigation graph | English/Hindi/Telugu | Val-Unseen / Test-Standard | NDTW、SDTW、SR、SPL |
| RxR-CE / RxR-Habitat | Habitat 连续空间，低层离散动作 | English/Hindi/Telugu | Val-Unseen / Test-Challenge | **NDTW**、SDTW、SR、SPL |

官方 RxR 说明中，RxR 使用 Test-Standard，而 RxR-Habitat 使用 Test-Challenge；两者
不是同一个测试集。RxR-Habitat 标准配置为30度转向/俯仰、0.25m前进和
480×640 RGB-D。RxR-Habitat leaderboard 按 NDTW 排名，而不是只按 SR 排名。

### Success Rate 定义

R2R-CE/RxR-CE 中通常在 agent 主动 `STOP` 后，若最终位置距目标不超过3m，则该
episode 成功。SR 是成功 episode 比例：

\[
\mathrm{SR}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[d_i\leq3\mathrm{m}].
\]

SPL 同时惩罚绕路；NDTW 衡量整条预测路径与参考路径的对齐程度。RxR 指令更长且
强调严格路径跟随，所以只看 SR 会漏掉“到达了终点但没有按指令走”的错误。

## R2R-CE 结果梳理

### 官方起点

VLN-CE 官方仓库给出的 CMA+PM+DA+Aug leaderboard baseline：

| Split | SR | SPL |
| --- | ---: | ---: |
| Val-Unseen | 29% | 27% |
| Test | 28% | 25% |

官方同时说明，论文曾报告 Val-Unseen SPL=30%，leaderboard 对同一模型重评为27%。
这提供了直接证据：不同 Habitat/硬件构建和评测入口会令论文数字与榜单数字不一致。

来源：[VLN-CE 官方仓库](https://github.com/jacobkrantz/VLN-CE#vln-ce-challenge-r2r-data)。

### 代表性连续环境进展

以下只整理可明确识别为 R2R-CE 的结果；`Val` 与 `Test` 不混排。

| 方法 | 年份/状态 | Val-Unseen SR | Val-Unseen SPL | Test SR | Test SPL | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CMA baseline | ECCV 2020 / 官方榜单 | 29 | 27 | 28 | 25 | 低层 recurrent baseline |
| ETPNav | TPAMI 2024 | 57 | 49 | 55 | 48 | evolving topological planning |
| HNR | CVPR 2024 | 61 | 51 | 约57 | 约49 | neural-radiance lookahead；不同表格有 rerun 版本 |
| NavMorph + ETPNav | ICCV 2025 | 59 | 50 | 57 | 49 | world-model self-evolution |
| NavMorph + HNR | ICCV 2025 | 64 | 53 | **60** | **52** | 当前推荐的 peer-reviewed Test 基准 |
| HSAN | NeurIPS 2025 | **64** | **59** | — | — | 只报告 Val-Unseen；不可称官方 Test SOTA |
| BrainNav/CompactNav | arXiv 2026-07 v1 | 约66 | 约54 | 63 | 55 | 最新 preprint claim，尚未独立核验 |
| StereoNav | arXiv 2026-05 | 81.1 | 68.3 | 未说明 | 未说明 | egocentric RGB claim；split/config 信息不足，不纳入最终基准 |

NavMorph 的 supplementary table 同时列出原论文行和带 `*` 的作者重跑行，说明同一
baseline 在不同实现中的数字不是完全相同。该文最终 HNR-backbone NavMorph 的
Test SR/SPL 为60/52，因此它比只引用 Val-Unseen 的方法更适合作为 NAV Test 目标。

来源：

- [NavMorph ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.html)
- [NavMorph supplementary](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Yao_NavMorph_A_Self-Evolving_ICCV_2025_supplemental.pdf)
- [HSAN NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/592da1445a51e54a3987958b5831948f-Paper-Conference.pdf)
- [BrainNav arXiv:2607.23181](https://arxiv.org/abs/2607.23181)
- [StereoNav arXiv:2605.13328](https://arxiv.org/abs/2605.13328)

### 为什么不把81.1%直接写成 R2R-CE 最终 SOTA

StereoNav 摘要给出 R2R-CE SR/SPL=81.1/68.3，但摘要没有明确说明是 Val-Unseen
还是 Test、是否使用标准 RGB-D/panoramic 配置、是否采用额外视觉先验以及是否提交
官方 leaderboard。它可以记录为“最新 self-reported claim”，但当前证据不足以替换
60/52这一可追溯的 peer-reviewed Test 数字。

同理，HSAN 的64/59是 Val-Unseen，不应与 NavMorph 的 Test 60/52直接比较高低。

## RxR-CE 结果梳理

### 官方挑战赛确定值

RxR-Habitat 2022 winner Reborn 的论文保存了 Test-Unseen/Test-Challenge leaderboard
表。完整关键结果为：

| 方法 | SR | SPL | NDTW | SDTW |
| --- | ---: | ---: | ---: | ---: |
| VLN-CE baseline | 13.93 | 11.96 | 30.86 | 11.01 |
| CWP-CMA | 24.08 | 19.07 | 37.39 | 18.65 |
| CWP-RecBERT | 24.85 | 19.61 | 37.30 | 19.05 |
| **Reborn** | **45.82** | **38.82** | **55.43** | **38.42** |

Reborn 是严格可称为“挑战赛 leaderboard winner”的结果。论文还说明官方主排序指标
是 NDTW；SR/SPL 是重要的次级指标。

来源：[RxR-Habitat 2022 winner paper](https://arxiv.org/abs/2206.11610)、
[RxR-Habitat 官方页面](https://ai.google.com/research/rxr/habitat)。

### 后续论文结果

| 方法 | 年份/状态 | Val-Unseen SR | Val-Unseen SPL | Test SR | Test SPL | Test NDTW |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reborn | 2022 challenge winner | 48.60 | 42.05 | 45.82 | 38.82 | 55.43 |
| ETPNav | TPAMI 2024 | 54.79 | 44.89 | 51.21 | 39.86 | 54.11 |
| HNR | CVPR 2024 | 56.39 | 46.73 | 53.22 | 41.14 | 55.61 |
| NavMorph + HNR | ICCV 2025 | 58.02 | 48.98 | **54.98** | **43.02** | **57.31** |
| HSAN | NeurIPS 2025 | 59 | 54 | — | — | — |
| MapDream | arXiv 2026 | 59.4 | 49.2 | — | — | — |
| BrainNav/CompactNav | arXiv 2026-07 v1 | 58.96 | 49.76 | — | — | — |

其中 HSAN 的 RxR-CE 表宣称 Val-Unseen SR/SPL=59/54，但没有 Test-Challenge 提交值；
它不能替换 Reborn 的“challenge winner”身份，也不能替换 NavMorph 的 Test 数字。

RxR 官方网页当前抓取结果显示 leaderboard 无条目，和2022 winner论文、官方回顾中的
历史结果冲突，推测是页面迁移或动态前端失效。不能把网页当前的“无条目”解释成历史
提交不存在。

来源：

- [RxR 官方数据与 split 说明](https://github.com/google-research-datasets/RxR)
- [RxR 官方 competition 说明](https://ai.google.com/research/rxr/explore)
- [Embodied AI Workshop 回顾](https://jiajunwu.com/papers/embodiedaiworkshop_arxiv.pdf)
- [MapDream arXiv:2602.00222](https://arxiv.org/abs/2602.00222)

## Qwen-RobotNav 专项追踪

### 模型、训练和开放状态

Qwen-RobotNav 是基于 **Qwen3-VL** 的端到端 waypoint policy。骨干后接轻量的
4-layer MLP action head，一次输出8个 `(x, y, theta)` waypoint；同一组权重统一处理
VLN、PointNav/ObjectNav、Tracking、Autonomous Driving 与 EQA。它不是显式建图模型，
核心是可由上层 agent 在推理时调节的 observation protocol：visual-token budget、
temporal decay、camera weight 与 frame-sampling mode。时间和相机身份通过
`Time step 0`、`Front View <image>` 等自然语言 tag 插入视觉 token 序列。

训练规模是 **15.6M samples**，batch 级配比为85% navigation trajectory planning 与
15% navigation-related vision-language reasoning；后者用于避免只训轨迹后退化为
reactive action mapper。模型从 Qwen3-VL 初始化并全参数训练；8B 使用 global batch
size 256，共 **2,816 H100 GPU-hours**。官方报告2B/4B/8B scaling，但 VLN 主表只列
4B/8B。官方仓库明确说明当前**没有公开 Qwen-RobotNav 权重的计划**，因此目前只能
引用结果，不能做 released-checkpoint reproduction。

来源：[Qwen-RobotNav Technical Report](https://arxiv.org/abs/2606.18112)、
[Qwen-RobotNav 官方仓库](https://github.com/QwenLM/Qwen-RobotNav)。

### 官方 Table 1：必须按传感器分组

以下全部是 VLN-CE **Val-Unseen**，数值按 Qwen-RobotNav 原文 Table 1 抄录。
`OS` 是 Oracle Success，RxR 同时给出 `nDTW`。表中的前人行是原文汇总值，不能视为
Qwen 团队统一重跑。

| Observation | 方法 | R2R NE↓ | R2R OS↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR nDTW↑ | RxR SR↑ | RxR SPL↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Monocular | NaVid | 5.72 | 49.2 | 41.9 | 36.5 | 5.72 | — | 45.7 | 38.2 |
| Monocular | Uni-NaVid | 5.58 | 53.3 | 47.0 | 42.7 | 6.24 | — | 48.7 | 40.9 |
| Monocular | NaVILA | 5.22 | 62.5 | 54.0 | 49.0 | 6.77 | 58.8 | 49.3 | 44.0 |
| Monocular | StreamVLN | 4.98 | 64.2 | 56.9 | 51.9 | 6.22 | 61.9 | 52.9 | 46.0 |
| Monocular | DualVLN | 4.05 | 70.7 | 64.3 | 58.5 | 4.58 | 70.0 | 61.4 | 51.8 |
| Monocular | InternVLA-N1 | 4.83 | 63.3 | 58.2 | 54.0 | 5.91 | 65.3 | 53.5 | 46.1 |
| Monocular | Qwen-RobotNav-4B | 4.22 | 73.6 | **66.9** | **60.5** | 4.15 | 68.6 | 71.3 | 61.5 |
| Monocular | Qwen-RobotNav-8B | 4.36 | 72.7 | 65.7 | 59.6 | 4.16 | 69.9 | **73.4** | **63.5** |
| Panoramic | NavFoM | 4.61 | 72.1 | 61.7 | 55.3 | 4.74 | 65.8 | 64.4 | 56.2 |
| Panoramic | ABot-N0 | 3.78 | 70.8 | 66.4 | 63.9 | 3.83 | — | 69.3 | 60.0 |
| Panoramic | OmniNav | 3.74 | 74.6 | 69.5 | 66.1 | 3.77 | — | 73.6 | 62.0 |
| Panoramic | AstraNav-World | 3.86 | 73.9 | 67.9 | 65.4 | 3.82 | — | 72.9 | 61.5 |
| Panoramic | Qwen-RobotNav-4B | 3.80 | 77.2 | 69.5 | 63.6 | 3.80 | 71.9 | 75.2 | 65.0 |
| Panoramic | Qwen-RobotNav-8B | **3.53** | **78.5** | **72.1** | **66.6** | **3.58** | **72.5** | **76.5** | **65.7** |

两个重要现象：R2R 单目 4B 反而略高于8B（66.9 vs. 65.7 SR），所以不能简单宣称
“参数越大每项越好”；RxR 长历史任务上8B提升更明确。其次，panoramic 的收益不能
全归因于网络设计，因为 observation information 本身更多。

### 引用与对比链

Qwen-RobotNav 的 VLN 表实际形成两条可比链。下面的“被比较工作”指各论文主要拿来
建立进步关系的代表方法，并非穷尽其 bibliography。

| 路线 | 工作 | 核心变化 | 其主要比较对象/承接关系 | Qwen 表中的位置 |
| --- | --- | --- | --- | --- |
| 单目 video policy | NaVid（RSS 2024） | RGB video + instruction 直接预测低层动作 | CMA、VLN-BERT、GridMM、Reborn、ETPNav 等 RGB-D/panoramic specialist | 单目起点 |
| 单目 unified VLA | Uni-NaVid（RSS 2025） | 3.6M samples，统一 VLN/ObjectNav/EQA/Tracking | NaVid、ETPNav 等单任务方法 | NaVid 后继 |
| 单目 legged VLA | NaVILA（RSS 2025） | VLM 高频动作解码与足式机器人部署 | HPN+DN、CMA、VLN-BERT、GridMM、Ego2-Map、DreamWalker、Reborn、ETPNav、NaVid | 单目54.0/49.3 SR |
| 单目 streaming VLA | StreamVLN（ICRA 2026） | SlowFast context，面向在线视觉流 | NaVid、Uni-NaVid、NaVILA 及传统 VLN-CE 方法 | 单目56.9/52.9 SR |
| 单目 dual system | DualVLN（2025） | 快速局部执行 + 慢速长程语义规划 | StreamVLN、InternVLA-N1 等 streaming/dual-system 方法 | Qwen 原文最强单目 R2R baseline |
| 单目/open dual system | InternVLA-N1（2025 Technical Report） | learned latent plan，System-2 + System-1 | ETPNav、NaVILA、NaVid，以及 NavDP/ShortestPathFollower 组合 | 单目与 RGB-D 两种配置均出现过 |
| Qwen 通用 VLA | Qwen-VLA（2025） | manipulation/navigation 跨任务统一；不是 RobotNav Table 1 baseline | 在 R2R/RxR 展示通用 VLA 能力，为 Qwen-RobotNav 的专用导航线提供前序证据 | 相关前身，不宜把其数字拼入 Table 1 |
| 多具身 foundation model | NavFoM（2025） | 8M samples，跨机器人/无人机/车辆及多导航任务 | Uni-NaVid、NaVILA、专用 VLN/Tracking/Driving 方法 | panoramic foundation baseline |
| 多任务 hierarchical VLA | ABot-N0（2026 Technical Report） | Brain-Action，16.9M trajectories + 5M reasoning samples | NavFoM、Uni-NaVid、NaVILA 等统一导航模型 | panoramic 强 baseline |
| fast-slow generalist | OmniNav（ICLR 2026） | prospective exploration + waypoint VLM，快慢系统 | NaVid、Uni-NaVid、NaVILA、InternVLA-N1、NavFoM 等 | Qwen 原文最强 panoramic baseline |
| world-model foresight | AstraNav-World（2026 preprint） | world model 用于 foresight control/consistency | OmniNav、NavFoM、ABot-N0 等新一代 generalist | panoramic 直接 baseline |
| agent-configurable foundation model | Qwen-RobotNav（2026 Technical Report） | 15.6M，多 task mode + 可调 observation protocol | 上述两条链，并扩展到 OVON/Tracking/Driving/EQA | 当前主题模型 |
| 最新单目后续 | Robostral Navigate（2026 preprint） | 8B、纯 monocular RGB、image-space pointing、episode packing + RL | Qwen-RobotNav、DualVLN、StreamVLN 等单目方法，也越级比较 multi-camera 系统 | 发表晚于 Qwen，非 Qwen 原表 |

对应的一层“祖先方法”可以简化为：

```text
CMA / VLN-BERT / GridMM / Reborn / ETPNav
                    ↓
NaVid → Uni-NaVid / NaVILA → StreamVLN / DualVLN / InternVLA-N1
                    ↓
          Qwen-RobotNav ← Qwen-VLA

Uni-NaVid / NaVILA → NavFoM → ABot-N0 / OmniNav / AstraNav-World
                                      ↓
                                Qwen-RobotNav
                                      ↓
                     Robostral Navigate（单目后续）
```

这不是严格的模型继承图，而是**论文对比与问题演化图**。例如 Qwen-RobotNav 建在
Qwen3-VL 上，并没有继承 NaVid 的权重；箭头表示后文把前文作为 baseline 或延续其
问题设定。

主要一手来源：

- [NaVid](https://arxiv.org/abs/2402.15852)
- [Uni-NaVid](https://arxiv.org/abs/2412.06224)
- [NaVILA 官方论文](https://navila-bot.github.io/static/navila_paper.pdf)
- [StreamVLN 官方仓库](https://github.com/InternRobotics/StreamVLN)
- [InternVLA-N1 / InternNav 官方仓库](https://github.com/InternRobotics/InternNav)
- [Qwen-VLA 官方仓库](https://github.com/QwenLM/Qwen-VLA)
- [NavFoM](https://arxiv.org/abs/2509.12129)
- [ABot-N0](https://arxiv.org/abs/2602.11598)
- [OmniNav](https://arxiv.org/abs/2510.06436)
- [AstraNav-World](https://arxiv.org/abs/2603.23745)
- [Robostral Navigate](https://arxiv.org/abs/2607.20785)

### 对 NAV 的直接启示

1. Qwen-RobotNav 的有效“历史”不是固定 KV cache，而是从视觉流中按 budget、recency
   和 camera weight 重新分配 token；这与 NAV register memory 的目标相邻，但不是
   同一种实现。后续 NAV 应在相同 token/显存预算下比较 register、uniform sampling
   与 recency sampling。
2. 15% V-L reasoning co-training 是很强的反例证据：纯 trajectory imitation 可能丢失
   通用语义和空间推理。NAV Stage Three 不宜只训练 action loss，应保留 language/
   spatial auxiliary supervision 或进行冻结/混训消融。
3. RobotNav 以8个 waypoint 统一不同 embodiment，说明从 Stage Two 的3D hidden state
   接轻量 waypoint/action head 是合理路线；但其性能不能证明 register 自身具备3D，
   NAV 仍需按现有设计做 layer probe 与3D supervision。
4. Robostral 的 episode packing 与 tree attention 把整条 episode 打包、同时阻止读取
   previous ground-truth actions，和 NAV 的多 chunk recurrent-register 训练非常相关，
   值得单独研究其并行训练是否能替代逐 chunk Python unroll。

## 论文是否直接引用别人的成功率

答案是：**普遍会，而且对 Test split 基本不可避免；但规范论文会通过引用、符号或
脚注明确来源。**

### 常见做法

1. **直接复制前人论文表格数字**
   
   Comparison with SOTA 中 baseline 行通常来自引用论文，不重新训练或评测。表格只
   能说明“文献报告值”，不能证明作者在统一代码环境复现成功。
2. **直接复制官方 leaderboard 数字**
   
   Test GT 被隐藏，自己的 Test 结果必须提交服务器；前人 Test 结果通常从榜单或前人
   论文复制。RxR 2022 winner 的 Table 3 就明确写为 leaderboard results。
3. **只重跑最接近的 backbone**
   
   新方法经常只重跑自己的 base model/ablation，较远的 baseline 继续引用原文。
4. **用符号区分来源**
   
   常见标记包括 `* reproduced by us`、`† panoramic`、`⋆ additional data`。例如
   MapNav 明确用 `*` 表示使用开源代码重现；NavMorph 同时列原始和 `*` rerun。
5. **Val-Unseen 也未必重跑**
   
   虽然 Val GT 可用，完整复现成本高且依赖 Habitat 版本，很多工作仍直接引用前人
   Val 数字。这正是不同论文表格中同一方法可能差1–3个百分点的原因之一。

### 对 NAV 写论文的要求

未来 NAV 的 comparison table 每个 baseline 行必须标注以下来源之一：

```text
Official leaderboard
Reported by original paper
Reproduced by us
Re-evaluated from released checkpoint
Self-reported preprint claim
```

不得把不同来源混在同一表格而不加注释。自己的 Val/Test 结果还要记录：

- dataset/version 与 split；
- Habitat-Sim/Habitat-Lab 版本；
- standard/panoramic/monocular camera；
- RGB 或 RGB-D；
- sliding/no-sliding；
- action step/turn angle/resolution；
- single model 或 ensemble；
- 是否使用 augmentation/额外数据；
- single run 或多次运行均值；
- checkpoint 和提交记录。

## NAV 最终采用的成功率表

### 主表：连续环境、Test 口径

| Benchmark | Baseline | SR | SPL | NDTW | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| R2R-CE Test | CMA official baseline | 28 | 25 | — | 最低官方基线 |
| R2R-CE Test | NavMorph + HNR | **60** | **52** | 约57 | 当前 peer-reviewed SOTA 目标 |
| RxR-CE Test-Challenge | Reborn official winner | **45.82** | **38.82** | **55.43** | 严格 challenge 对比 |
| RxR-CE Test | NavMorph + HNR | **54.98** | **43.02** | **57.31** | 当前后续 peer-reviewed Test 目标 |

### 辅表：Val-Unseen 研究追踪

| Benchmark | 可靠已发表高值 | 最新 preprint claim | 使用规则 |
| --- | --- | --- | --- |
| R2R-CE Val-Unseen | HSAN 64/59 SR/SPL | Qwen-RobotNav panoramic 8B 72.1/66.6；Robostral monocular 77.4 SR；StereoNav 81.1/68.3口径不清 | 按 sensor 分组，不替代 Test 主表 |
| RxR-CE Val-Unseen | HSAN 59/54 SR/SPL | Qwen-RobotNav panoramic 8B 76.5/65.7、monocular 8B 73.4/63.5；Robostral monocular 75.1 SR | 同时报告 NDTW/SDTW，按 sensor 分组 |

### NAV 的阶段性目标

| 等级 | R2R-CE SR | RxR-CE SR | 解释 |
| --- | ---: | ---: | --- |
| 可用 baseline | 约45% | 约35% | 已超过早期 recurrent baseline |
| 强基线 | 55% | 45% | 接近 ETPNav/Reborn 水平 |
| 论文竞争线 | 60% | 55% | 接近 peer-reviewed Test 高值 |
| 最新 claim 追踪线 | 72%+ | 75%+ | 对应2026 Val-Unseen foundation-model claim；必须统一 sensor 口径 |

这些数值是对**完整 benchmark split 的 episode-level SR**，不是训练集准确率、动作
预测 accuracy，也不是少量 episode smoke test。

## 限制与未决问题

1. EvalAI R2R 页面和 Google RxR 动态 leaderboard 当前无法稳定导出完整历史条目；
   文档优先采用官方仓库、winner paper 和同行评审论文中可追溯的表格。
2. 2026年部分工作仍是 arXiv v1，数字尚未经过同行评审或独立复现，不能直接更新
   NAV 主表。
3. panoramic、monocular、egocentric RGB、RGB-D、额外数据和 ensemble 会显著改变
   难度；未来若 NAV 使用不同 sensor，不应直接宣称超越标准 leaderboard。
4. SR 不能替代 RxR-CE 的 NDTW。NAV 在 RxR-CE 上必须至少同时报告
   `SR/SPL/NDTW/SDTW`。
5. 同一方法可能有原论文值、作者重跑值、第三方重跑值和 leaderboard 值；最终投稿
   前应重新冻结一版带来源列的 comparison table。
6. Qwen-RobotNav、ABot-N0 等 Technical Report 的发布日期、版本与表格仍可能更新；
   本文快照对应2026-08-01，投稿前需按 arXiv version 再核验。

## 相关资源

- 本地 R2R episodes：`/sharedata/datasets/R2R`
- 本地 MP3D scenes：`/sharedata/datasets/mp3d/v1/tasks/mp3d`
- 复现协议：`../05_evaluation/r2r_ce_sota_reproduction.md`
- 资源总账：`../06_operations/resource_inventory.md`
- 数据准备规范：`../03_data/dataset_preparation_specification.md`
- Stage Three 导航设计：`../01_design/v0_stage_three_navigation.md`
