# NAV 评测协议、外部复现与 Benchmark 调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-010` |
| 类型 | 评测、复现与 Benchmark 总览 |
| 状态 | Live / Reference |
| 更新时间 | 2026-08-15 |
| 职责 | 集中维护 VBench/iWorldBench/R2R-CE/RxR-CE 等评测协议、旧 InfiniteWorld 对照、外部 WAM/VLA 复现结果和 SOTA 调研。 |

## 当前入口结论

本文件只记录可复现协议、外部 baseline 和 benchmark 口径。概念性结论必须标注已验证/未验证；scaffold/toy/small diagnostic 不得作为正式验收证据。

## 合并主题

- V1 完整模型链路 Smoke 记录
- V0 InfiniteWorld VBench 技术指标子集评测
- R2R-CE / RxR-CE SOTA 复现协议
- BridgeVLA++ memoryBench 复现记录
- Fast-WAM 与 GigaWorld-Policy 复现记录
- R2R / RxR Benchmark 与最新 SOTA 调研
- 世界模型训练算力与速度调研
- Representation-only WAM 与 Video-Action Co-training 调研

## 维护规则

- 后续相关主题优先更新本文档，不再新增细碎同主题文档。
- 历史 run、旧结构和 superseded 方案保留在对应章节中，用于追溯和对照。
- 若代码/日志/文档三线不一致，以代码与真实记录为准先修正文档。

## 合并正文


## V1 完整模型链路 Smoke 记录


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-005` |
| 类型 | 完整模型链路验证（Full-pipeline Smoke） |
| 状态 | Historical Smoke / Superseded by DEC-041 |
| 更新时间 | 2026-08-14 |
| 职责 | 记录新版 V1 完整模型在三阶段训练、参数更新、结构约束和两种推理模式上的最小可运行验证 |

### 当前结论

该 smoke 在 DEC-041 前验证了当时代码链路可运行，但它使用
`shared dual-stream backbone`，因此不再作为正式 shared WanBlock 结构验收。
当前记录只保留为历史 diagnostic。待代码改成 single shared WanBlock token stream
后，必须重新跑完整 Stage1/2/3 smoke。

历史 smoke 入口是：

```bash
bash NAV/scripts/run_v1_full_pipeline_smoke.sh cpu
```

或指定 CUDA：

```bash
CUDA_VISIBLE_DEVICES=0 bash NAV/scripts/run_v1_full_pipeline_smoke.sh cuda:0
```

该 smoke 不再使用旧 scaffold、小旁路 head、旧 A/B Register 或旧
InfiniteWorld adapter。参数可以随机初始化，但必须走当前完整结构：

```text
R_null fixed template
  -> RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  -> historical shared dual-stream backbone
      visual stream: Register / obs / future noisy video
      action stream: A_noise
  -> video generation head
  -> action flow decoder
  -> 3D geometry probe
```

### 覆盖范围

| 检查项 | 要求 |
| --- | --- |
| Stage One | `L_visual_flow` 正常 forward/backward/update；`A_cur`、`A_noise` 保留格式但不计算 action supervision |
| Stage Two | 同一视频生成前向上增加 `L_3D`，确认 `geometry_probe` 和 backbone 有梯度更新；DEC-041 后需用 single shared WanBlock 重跑 |
| Stage Three | `L_action_flow + CE_aux` 为主，同时加入 video/3D rehearsal loss，避免 backbone/Register 退化 |
| Videogen inference | 输出 `z_future` 与 `future_velocity`，shape 与 noisy future latent 对齐 |
| Policy inference | 不输入 future noisy video，输出 `action_chunk`、`primitive_logits`、`primitive_ids` |
| 结构审计 | 必须使用 `RegisterCell`；不得出现 RegisterExtractor/RegisterUpdater；不得出现 action/latent additive bias |

### 2026-08-14 运行记录

CPU 完整链路：

```text
report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json

device:
  cpu

inference:
  videogen_z_future       = [2, 16, 2, 8, 8]
  videogen_future_velocity= [2, 16, 2, 8, 8]
  policy_action_chunk     = [2, 10, 6]
  policy_primitive_logits = [2, 10, 12]
  policy_primitive_ids    = [2, 10]
```

清理后的 CUDA smoke 也已通过：

```text
report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014807/report.json

device:
  cuda:0
```

后续每次改动模型结构、attention 规则、loss 组成、action/token 接口或训练入口后，
必须重新跑该 full-pipeline smoke，并把 report 路径登记到本文或对应实验记录中。

### 旧入口清理

2026-08-14 已从当前代码主线移除旧 scaffold、旧 `A_query`、旧 action-bias、
旧 InfiniteWorld adapter 和旧 A/B Register 训练入口。V0 论文/实验结果文档仍可
作为历史解释，但不再对应可执行主线脚本。若需要复现 V0，请从 git 历史恢复到
清理前 commit，而不是把旧脚本混回 V1 主线。


## V0 Infinite-World 的 VBench 技术指标子集评测


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-001` |
| 类型 | 评测协议与记录（Evaluation Protocol） |
| 状态 | Historical Baseline / 旧入口已从当前代码删除 |
| 更新时间 | 2026-08-06 |
| 职责 | 固定 VBench 对齐口径并按时间记录每次测评（模型/训练实现 + 测评条件 + 结果） |

---

### 0. 评测协议与口径（所有记录共用）

本文件保留 V0 InfiniteWorld/Register A/B 的 VBench 复现记录。文中的旧 NAV
Register 推理脚本已从当前代码主线删除；当前 V1 结构验证见
`evaluation_reproduction_and_benchmarks.md`。

#### 0.1 VBench 指标

- **六指标版**（仅记录 1 用）：`temporal_flickering`、`dynamic_degree`、
  `motion_smoothness`、`imaging_quality`、`subject_consistency`、
  `background_consistency`。汇总分为 VBench 归一化 `[0,1]`；单视频成像质量
  采用未归一化 MUSIQ 原始分，其汇总分由 VBench 归一化。
- **四指标版**（论文协议，记录 2 起默认）：`motion_smoothness`、
  `dynamic_degree`、`aesthetic_quality`、`imaging_quality`；**Average Score =
  四项算术平均**；明确排除 `subject_consistency`/`background_consistency`
  （交互探索含大幅视角切换，主体与背景本身非平稳）。

#### 0.2 论文评测协议（核对自原文 4.2.1 / Table 1 / 4.2.3）

原论文 *Infinite-World: Scaling Interactive World Models to 1000-Frame
Horizons via Pose-Free Hierarchical Memory*（arXiv:2602.02393v2）：

- 100 个 Gemini 文本提示词（Indoor/Street/Nature/Fantasy）× 10 条人工
  长动作轨迹（**每条 16 chunks**），每场景分配一条轨迹；
- 客观指标 = VBench 四项，Average Score = 四项算术平均；
- 16 chunks → 1 + 16×80 = 1281 帧（约 42.7 秒）；
- 论文 Infinite-World 成绩：MS 0.9876 / DD 1.0000 / AQ 0.5440 /
  IQ 0.7159 / 平均 0.8119；
- 长历史"专属"指标是 User Study 三维度（Memory Consistency / Visual
  Fidelity / Action Responsiveness）+ ELO，VBench 四项只衡量通用质量、不
  直接度量长程记忆/loop-closure。

#### 0.3 通用生成与评测设置

| 项 | 配置 |
| --- | --- |
| Infinite-World 代码版本 | `1e8fc49`（本地 env-var 化、未改核心生成逻辑） |
| VBench 代码版本 | `45e79ec` |
| Infinite-World 权重 | `/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt` |
| Wan VAE / UMT5 权重 | `/sharedata/Wan2.1-T2V-1.3B` |
| 随机种子 | `42` |
| 采样步数 | `30` |
| CFG | `5.0` |
| 生成格式 | H.264、896×448、30 FPS；N chunks → 1+N×80 帧 |
| 评测模式 | VBench `custom_input`，`--load_ckpt_from_local True` |

#### 0.4 评测命令模板

Infinite-World 推理（官方脚本，本地 env-var 化）：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
INFWORLD_MAX_PROMPTS=1 INFWORLD_NUM_CHUNKS=<N> INFWORLD_SAMPLING_STEPS=30 \
INFWORLD_SEED=42 INFWORLD_CFG_SCALE=5.0 \
INFWORLD_OUTPUT_DIR=<out> INFWORLD_FILENAME_SUFFIX="_seed0042" bash infer_local.sh 1
```

NAV A/B 推理（Register 版）：

```bash
python NAV/scripts/infer_register_vbench.py --variant <latent_prefix|dit_condition> \
  --checkpoint <ckpt.pt> --output-dir <out> --device cuda:0 \
  --chunks <N> --steps 30 --cfg-scale 5.0 --prompt-indices 0 --seeds 42
```

VBench 评测：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/VBench && source config/sharedata.env
export PATH="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_vbench/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --videos_path <videos> \
  --dimension motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
  --mode custom_input --load_ckpt_from_local True --output_path <metrics>
```

#### 0.5 测试输入与逐 chunk 规则（NAV Register 版 / InfiniteWorld HPMC 版）

本节明确每次推理喂给模型的四类输入——**文本 text、local memory（那一帧）、
长程 history、action**——以及单 chunk 与 3-chunk 测试在规则上的区别。两套推理
脚本共用同一份 demo 条件，但 history 路径不同。

##### 0.5.1 四类输入的定义

来源：`Infinite-World/prompts/demo.yaml[prompt_index]`，每条 =
`[prompt, condition_image_path, action_json_path]`。当前所有记录用
`prompt_index=0`（campus walkway）。

1. **文本 text（prompt）**：`demo.yaml` 中的自然语言描述字符串，例如
   "A serene campus walkway lined with modern glass buildings, green ivy
   climbing some walls, empty benches, soft dappled sunlight through maple
   trees."。由 UMT5/T5 编码为文本 condition。负向 prompt 固定为脚本内
   `NEGATIVE_PROMPT`（抑制静止/低质/拥挤等）。CFG guidance = 5.0，正/负 prompt
   合成 batch=2 一起前向。

2. **local memory（local_latent，"那一帧"）**：**最近一帧的 VAE latent**，
   即 `history_latent[:, :, -1:]`（时间维取最后一帧，1 个 latent frame）。
   - chunk 0：条件图 `0001.jpg` 经 VAE encode 后的最后一帧 latent；
   - chunk ≥1：上一 chunk 生成 latent `samples` 的最后一帧。
   它是"紧邻的即时过去帧"，给模型提供本 chunk 起点的局部连续性。在 NAV 脚本中
   显式以 `local_latent` 传入；InfiniteWorld 官方脚本中由 `image_cond`（完整
   历史 latent）隐式承载，最后一帧即在其中。

3. **长程 history**：跨 chunk 的远端记忆，两套脚本路径不同：
   - **NAV Register 版**（`infer_register_vbench.py`，记录 3/4/7）：HPMC 已
     `remove_hmpc()`，长程历史**只走 Register**：
     - chunk 0：`register_memory.extract(history_latent)` 从条件 chunk 直接
       建立固定预算 Register；
     - chunk ≥1：`register_memory.update(registers, history_latent)` 用本 chunk
       生成 latent 递归更新 Register（固定预算、不增长）。
     - 传入的 `image_cond` 被 `CFGRegisterWrapper` 丢弃，**长程历史唯一通道是
       Register**，不重 encode 像素 buffer。
   - **InfiniteWorld HPMC 版**（`infworld_inference.py`，记录 1/2/5/6）：长程
     历史走 HPMC——每 chunk 把累积的 `video_buffer`（像素）VAE encode 成
     `image_cond`，再由 DiT 内 HPMC 压成固定预算记忆 token。缓存版（记录 6）改为
     在 latent 域累积 `latent_buffer`，省去全 buffer 重 encode。

4. **action（move + view）**：`0001.json` 是逐帧离散动作序列，每帧一项
   `{"move": ..., "view": ...}`，映射到 0–9 的类别索引（`MOVE_ACTION_MAP` /
   `VIEW_ACTION_MAP`，含 no-op / go forward / turn left ... / uncertain）。
   - 每个 chunk 对应 `num_frames`（=81 像素帧）个动作步；按当前 chunk 在整段
     中的位置 `[start:end]` 切片，末尾不足则用 0（no-op）补齐。
   - move、view 两个 `[num_frames]` 长整型向量，`repeat(2,1)` 成 batch=2 供 CFG。
   - action 是**逐帧**的（每像素帧一个动作），不是每 chunk 一个。

##### 0.5.2 单 chunk 与 3-chunk 的规则区别

设 `N = chunks`。输出帧数 = `1 + N×80`（首帧为条件图，每 chunk 丢掉与上段重叠
的首帧后接 80 新帧）。

| 维度 | 单 chunk（N=1，记录 1/2，81 帧） | 3-chunk（N=3，记录 5/7，241 帧） |
| --- | --- | --- |
| text | 同一条 prompt，全程不变 | 同一条 prompt，全程不变 |
| local memory | 条件图最后一帧 latent | chunk0=条件图末帧；chunk1/2=上一生成 chunk 末帧（滚动） |
| 长程 history（NAV） | **仅 extract**，Register 只编码条件图，**无 update、无跨 chunk 累积** | extract + 2 次 update，Register 跨 3 chunk 递归累积长程记忆 |
| 长程 history（InfWorld） | HPMC 只压条件 latent | HPMC 每 chunk 压一次不断增长的 history（或缓存 latent 版累积） |
| action | 81 步（move+view），取 `[0:81]` | 3×81=243 步，chunk0 `[0:81]`、chunk1 `[81:162]`、chunk2 `[162:243]`，末尾不足补 0 |
| 生成难度 | 单段、无长程反馈 | 第 2/3 chunk 须依赖 Register/HPMC 长程记忆保持一致，难度更高 |
| DD 风险 | 低（短段不易塌缩） | 高（长段更易出现静止/重复，DD 易转 false，见记录 4/7） |

**关键区别总结**：单 chunk 只测"给定一帧 + 一段动作能生成多好的一段"，长程
history 通道**几乎不被使用**（NAV 只 extract 不 update，HPMC 只压单段）；
3-chunk 才真正启用长程记忆——NAV 的 Register 要经 2 次 update 跨 chunk 累积，
InfiniteWorld 的 HPMC 要压不断增长的历史——因此 3-chunk 是检验长程记忆是否
有效的最小有意义长度，DD/一致性退化也主要在此暴露。当前所有 3-chunk 记录
仍仅 1 场景 1 seed，统计意义有限，扩到多场景多 seed 后才能对 A vs B 的长程
动态维持能力下统计结论。

#### 0.6 `result/vbench/` 目录命名规则

自 2026-08-06 起，`result/vbench/` 下每次测评目录统一命名为：

```text
<测试日>_<模型名>_<测试bench>_<chunk历史>_<历史范式>
```

各字段定义（分隔符统一为下划线 `_`）：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| 测试日 | `YYYYMMDD` | 测评运行日期，如 `20260805` |
| 模型名 | `navA` / `navB` / `infworld` / `threeway` | 模型身份；可附训练步数后缀 `_s<N>`，如 `navA_s500`、`navB_s1000`；`infworld` = InfiniteWorld 官方权重基线；`threeway` = 多模型对比汇总 |
| 测试bench | `vbench` | 评测基准，目前只有 `vbench` |
| chunk历史 | `<N>chunk` | **生成 chunk 数** N（输出帧 = 1 + N×80），如 `1chunk`/`2chunk`/`3chunk`/`16chunk` |
| 历史范式 | `autoreg` / `gthist` | `autoreg` = 自回归回灌（生成 latent 回灌作历史）；`gthist` = 全部真实历史（teacher-forced，给 n 个 GT 历史 chunk 预测 1 个新 chunk） |
| 样本配置（可选第 6 字段） | `stats<N>` / `scn<M>_sd<K>` | 多样本统计运行追加，如 `_stats10`（2 场景×5 种子）、`_stats20`（2 场景×10 种子）；单样本运行省略 |

示例：

- `20260805_navA_s500_vbench_3chunk_autoreg` —— 2026-08-05、NAV A 500 步、VBench、3 生成 chunk、自回归
- `20260805_navB_s1000_vbench_3chunk_autoreg` —— 2026-08-05、NAV B 1000 步、VBench、3 生成 chunk、自回归
- `20260805_infworld_vbench_16chunk_autoreg` —— InfiniteWorld 官方权重、16 chunk、自回归（对标论文 Table 1）
- `20260806_navA_s500_vbench_3chunk_gthist` —— NAV A 500 步、3 chunk、全部真实历史（teacher-forced 诊断）

规则与约束：

1. **历史范式二选一**：`autoreg` 与 `gthist` 是互斥的两条测评线——前者对标
   InfiniteWorld/LingBot 论文的自回归 rollout 协议（可跨模型比），后者是 NAV
   内部训推一致诊断（给 GT 历史测单步预测，不与外部模型比）。
2. **chunk 数指生成 chunk 数**，不含 1 帧条件种子；`3chunk` = 241 帧。
3. **模型名带步数后缀**时用 `_s<N>`，避免与字段分隔符混淆；无后缀默认最终/官方权重。
4. **新测评一律用本规则命名**；2026-08-06 已对历史目录统一改名（见文末
   "改名映射表"），仅 `infiniteworld_prior_runs`（多条早期小复现的合集，无法
   映射到单一名称）与 `frame_analysis`（跨模型帧级分析，非单次测评）作为
   豁免遗留目录保留原名。
5. 目录内子结构沿用 `videos/`、`metrics/`、`logs/`（+ teacher-forced 线另加
   `inputs/`、`gt/`）。

---

### 测评记录

#### 记录 1 — InfiniteWorld 官方权重 6 指标基线（2026-07-23）

- **时间**：2026-07-23
- **模型实现与训练实现**：InfiniteWorld 官方权重
  `infinite_world_model.ckpt`（非 NAV 训练，纯基线）；推理用官方
  `scripts/infworld_inference.py`（本地 env-var 化，核心逻辑未改）。
- **测评条件**：2 个官方演示条件（campus + fantasy city），1 chunk，81 帧；
  seed 42，30 步，CFG 5.0；VBench **六指标**，`custom_input`。
- **测评结果**：

| 指标 | 汇总分 | Campus | Fantasy City |
| --- | ---: | ---: | ---: |
| Temporal Flickering | 0.949990 | 0.965092 | 0.934888 |
| Dynamic Degree | 1.000000 | true | true |
| Motion Smoothness | 0.982819 | 0.983717 | 0.981920 |
| Imaging Quality | 0.726528 | 79.0273（原始） | 66.2783（原始） |
| Subject Consistency | 0.956838 | 0.982987 | 0.930690 |
| Background Consistency | 0.975151 | 0.989154 | 0.961148 |

- **产物路径**：
  - 视频：`Infinite-World/outputs/infworld-ckpt0-step30-cfg5.0/000{0,1}_*.mp4`
  - 结果：`NAV/result/vbench/infiniteworld_prior_runs/metrics/infinite_world_30step_subset/results_2026-07-23-22:15:19_eval_results.json`
- **解读与局限**：两段均通过动态运动测试，高一致性非静止输出；成像质量最低；
  fantasy city 在闪烁/成像/主背景一致性上均弱于 campus。仅 2 个官方样本，
  统计不确定性大，宜作后续 NAV 回归测试基线。Infinite-World 需初始图+动作，
  无法直接跑完整 VBench T2V。

#### 记录 2 — InfiniteWorld 官方权重 论文四项协议小规模复现（2026-07-23）

- **时间**：2026-07-23
- **模型实现与训练实现**：同记录 1（InfiniteWorld 官方权重，非 NAV 训练）。
- **测评条件**：
  - (a) 在记录 1 的两个官方 1-chunk、81 帧视频上补测美学质量，按论文四项公式；
  - (b) 单 campus 场景生成 2 chunks（161 帧，5.37 秒），seed 42，30 步，CFG 5.0；
  - 评测 VBench 四项，Average Score = 四项算术平均。
- **测评结果**：

(a) 两个官方 1-chunk：

| 指标 | 本地汇总分 | 论文分数 |
| --- | ---: | ---: |
| Motion Smoothness | 0.982819 | 0.9876 |
| Dynamic Degree | 1.000000 | 1.0000 |
| Aesthetic Quality | 0.643381 | 0.5440 |
| Imaging Quality | 0.726528 | 0.7159 |
| 四项平均 | 0.838182 | 0.8119 |

(b) 单个 2-chunk campus：

| 指标 | 2-chunk 本地分数 | 论文分数 | 差值 |
| --- | ---: | ---: | ---: |
| Motion Smoothness | 0.978976 | 0.9876 | -0.008624 |
| Dynamic Degree | 1.000000 | 1.0000 | 0.000000 |
| Aesthetic Quality | 0.608450 | 0.5440 | +0.064450 |
| Imaging Quality | 0.782783 | 0.7159 | +0.066883 |
| 四项平均 | 0.842552 | 0.8119 | +0.030652 |

- **产物路径**：
  - 1-chunk 美学：`NAV/result/vbench/infiniteworld_prior_runs/metrics/infinite_world_30step_paper_protocol_small/results_2026-07-23-22:40:17_eval_results.json`
  - 2-chunk 视频：`NAV/result/vbench/20260723_infworld_vbench_2chunk_autoreg/videos/0000_A_serene_campus_walkway_lined_.mp4`
  - 2-chunk 结果：`NAV/result/vbench/20260723_infworld_vbench_2chunk_autoreg/metrics/results_2026-07-23-22:54:10_eval_results.json`
- **可比性结论**：已复现论文四项指标、平均分公式、30-step、跨 chunk 滚动机制，
  但未复现数据规模。本地分高于论文平均，主要因仅用 1 个较易 campus 场景，
  不能解释为本地优于论文。论文 100 初始图 + 10 条人工轨迹未在官方仓库公开。

#### 记录 3 — NAV A/B 全参 step-1000 同协议对比（2026-07-25）

- **时间**：2026-07-25
- **模型实现与训练实现**：A/B 各自训练至 step 1000 的 `full-final.pt`。
  - 变体：A = `latent_prefix`（固定 `T=4` 的 spatial Register latent prefix）；
    B = `dit_condition`（16 个 Register token 与文本 condition 拼接）。
  - 训练：从 InfiniteWorld 初始化，RE10K 全量，`full-ebs4`（effective batch
    size 4），1000 步；Extractor/Updater 随机初始化。
  - 推理：NAV `scripts/infer_register_vbench.py`；只保留单帧 local memory，
    长期 history 不经 HPMC，每生成完一个 chunk 用该 chunk 更新一次 Register。
- **测评条件**：单 campus 2-chunk（161 帧），seed 42，30 步，CFG 5.0；
  同一份本地 VBench 权重，VBench 四项，四项算术平均。与记录 2(b) 同口径。
- **测评结果**：

| 模型 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Imaging Quality | 四项平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| InfiniteWorld 本地基线（记录2b） | 0.978976 | 1.000000 | 0.608450 | 0.782783 | 0.842552 |
| A：latent prefix @1000 | **0.979322** | **1.000000** | **0.635929** | **0.786550** | **0.850450** |
| B：DiT condition @1000 | 0.982449 | 0.000000 | 0.635351 | 0.783273 | 0.600268 |

A 相对 InfiniteWorld 基线：MS +0.000346 / DD +0.000000 / AQ +0.027479 /
IQ +0.003766 / 四项平均 +0.007898。

- **产物路径**：
  - 视频：`NAV/result/vbench/20260725_navA_s1000_vbench_2chunk_autoreg/videos/0000_*.mp4`、`20260725_navB_s1000_vbench_2chunk_autoreg/videos/0000_*.mp4`
  - 结果：`20260725_navA_s1000_vbench_2chunk_autoreg/metrics/results_2026-07-25-13:19:26_eval_results.json`、`20260725_navB_s1000_vbench_2chunk_autoreg/metrics/results_2026-07-25-13:20:47_eval_results.json`
  - 日志：`20260725_navA_s1000_vbench_2chunk_autoreg/logs/vbench_a_{inference,evaluation}.log`、`20260725_navB_s1000_vbench_2chunk_autoreg/logs/vbench_b_{inference,evaluation}.log`
- **解读与局限**：B 的 MS/AQ/IQ 未明显恶化但 DD=false（"画面平滑但有效运动
  不足"）。单样本 DD 是 0/1 二值，会使四项平均产生 −0.25 量级跳变，是优先
  排查信号但不能作 B 的统计结论。单样本结果支持优先保留 A（四项平均比基线
  高约 0.79 个百分点）。下一步应扩展到 8–16 场景，统计 B 的 DD 通过率。

#### 记录 4 — NAV A/B from-scratch 中间 checkpoint 同协议对比（2026-08-04）

- **时间**：2026-08-04
- **模型实现与训练实现**：Stage One 1.0 按 DEC-019 重启为 from-scratch 训练。
  - 变体：A = `latent_prefix`（GPU0）、B = `dit_condition`（GPU1）。
  - 训练：from-scratch，effective batch size 16（micro×accum），1000
    effective optimizer steps，DL3DV full-episode latent，empty UMT5；
    Extractor/Updater 随机初始化，不加载任何旧 checkpoint。
  - 本次 checkpoint：A = `full-step-000500.pt`（实际训练到 step 541，最后
    ckpt@500，已按用户指示停止，后续从 500 续训）；B = `full-step-000800.pt`
    （实际 step 759+，ckpt@800）。
- **测评条件**：同记录 3（单 campus 2-chunk，161 帧，seed 42，30 步，CFG 5.0，
  VBench 四项，四项算术平均）。
- **测评结果**：

| 模型 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Imaging Quality | 四项平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| InfiniteWorld 本地基线（记录2b） | 0.978976 | 1.000000 | 0.608450 | 0.782783 | 0.842552 |
| A@1000（记录3） | 0.979322 | 1.000000 | 0.635929 | 0.786550 | 0.850450 |
| B@1000（记录3） | 0.982449 | 0.000000 | 0.635351 | 0.783273 | 0.600268 |
| **A@500（本次）** | **0.983811** | **1.000000** | **0.617231** | **0.782360** | **0.845850** |
| **B@800（本次）** | **0.979446** | **1.000000** | **0.639200** | **0.778224** | **0.849217** |

- **工程说明（一处必要修复）**：本次 A 推理首次报 `float != bfloat16`
  dtype 错误，根因是 `src/nav/spatial_register_memory.py` 的 `extract`
  把池化结果 cast 到 `chunk_latent.dtype`（fp32），而 Register 权重是 bf16。
  已改为 cast 到 block 权重 dtype
  （`self.extract_blocks[0].register_in.weight.dtype`）。训练路径输入已是
  bf16，该改动对训练是 no-op，仅修复 fp32 VAE latent 进入 Register 的 eval
  路径。B（`dit_condition`）走另一套 Register 模块，未受影响。
- **产物路径**：
  - 视频：`NAV/result/vbench/20260804_navA_s500_vbench_2chunk_autoreg/videos/0000_seed0042_*.mp4`、`20260804_navB_s800_vbench_2chunk_autoreg/videos/0000_seed0042_*.mp4`
  - 结果：`20260804_navA_s500_vbench_2chunk_autoreg/metrics/*_eval_results.json`、`20260804_navB_s800_vbench_2chunk_autoreg/metrics/*_eval_results.json`
  - 日志：各目录下 `logs/inference.log`、`logs/evaluation.log`
- **解读与局限**：B@800 通过 DD，四项平均 0.8492 为当前所有样本最高，回应
  了记录 3 中 B@1000 DD=false 的警戒（单样本偶然静止而非固有退化）。A@500
  相对 A@1000 四项平均略降（0.8505→0.8459）但仍高于基线，半训练步数质量
  基本守住。B@800 在 AQ 上领先 A@500（+0.0220），四项平均反超。仍为单样本
  口径，统计意义有限，应扩展到 8–16 场景（可复用
  `scripts/run_vbench_threeway_stats20_gpu0.sh` 的 20 样本流程）。

#### 记录 5 — InfiniteWorld 官方权重 3-chunk 复现（2026-08-05，已完成）

- **时间**：2026-08-05
- **模型实现与训练实现**：InfiniteWorld 官方权重（非 NAV 训练）；推理用官方
  `scripts/infworld_inference.py`（每 chunk 重 encode 整个 video buffer 为
  `image_cond`，HPMC 在 DiT 内部压缩历史 latent 至固定预算 T_out=20）。
- **测评条件**：单 campus 3-chunk（241 帧，约 8.03 秒），seed 42，30 步，
  CFG 5.0，`expandable_segments=True`；VBench 四项，四项算术平均。与记录 2b
  同口径但 horizon 更长。
- **测评结果**：

| 模型 | chunks / 帧数 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Imaging Quality | 四项平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InfiniteWorld（记录2b） | 2 / 161 | 0.978976 | 1.000000 | 0.608450 | 0.782783 | 0.842552 |
| **InfiniteWorld 3-chunk（本次）** | **3 / 241** | **0.979444** | **1.000000** | **0.581495** | **0.788454** | **0.837348** |

- **产物路径**：
  - 视频：`NAV/result/vbench/20260805_infworld_vbench_3chunk_autoreg/videos/0000_seed0042_*.mp4`
  - 结果：`NAV/result/vbench/20260805_infworld_vbench_3chunk_autoreg/metrics/*_eval_results.json`
  - 日志：`NAV/result/vbench/20260805_infworld_vbench_3chunk_autoreg/logs/{inference,evaluation}.log`
- **推理过程**：前两次因显存不足失败（GPU0 与另一用户 `starry` 的
  `vlm_server.py` 共享、可用仅 ~31 GiB，每 chunk 重 encode 整个 buffer）；
  第三次 `vlm_server.py` 退出后 GPU0 全空，3 chunks 全部生成成功。
- **解读与局限**：3-chunk 相对 2-chunk：MS 持平、DD 维持通过、IQ 略升
  （+0.0057），**AQ 明显下降（−0.0270）**，四项平均 −0.0052。与"滚动越长、
  误差累积越明显"一致。该 3-chunk 基线可作为 NAV A/B 3-chunk-as-history
  对比的直接参照。仍为单样本口径；Infinite-World HPMC 与 NAV Register 历史
  机制不同，数值仅作同口径参照。

#### 记录 6 — InfiniteWorld 16-chunk 论文标准复现（2026-08-05，进行中）

- **时间**：2026-08-05
- **模型实现与训练实现**：InfiniteWorld 官方权重；推理用官方
  `scripts/infworld_inference.py`。原脚本每 chunk 重 encode 整个像素 buffer，
  47 GB 卡撑不到 16 chunk（完成到 chunk 4，显存与步时线性增长：chunk1
  29 GB/6.6 s·step⁻¹ → chunk5 48 GB/18 s·step⁻¹，HPMC 在该路径未压住
  VAE 全 buffer encode）。已改为**缓存 latent 版**：累积生成 latent
  `samples[:,:,1:]` 作历史，不再重 encode 整个 buffer（VAE 严格因果
  `CausalConv3d`/`CACHE_T=2`，latent 前缀稳定，与论文 latent-domain HPMC
  等价，仅省去 decode→encode 往返）；并改 CPU 保存 + `.pt` 备份防保存 OOM。
- **测评条件**：单 campus 16-chunk（1281 帧，约 42.7 秒），seed 42，30 步，
  CFG 5.0；VBench 四项，四项算术平均。对齐论文 Table 1 的 chunk 长度（但
  仅 1 场景 vs 论文 100 场景）。
- **测评结果**：进行中。
  - 原脚本：失败（OOM@chunk5，无视频产出）。
  - 缓存版首次：16 chunk 全部生成成功（1281 帧，HPMC Case B 平台期，步时
    稳定 ~18 s、显存稳 ~47.8 GB），但最后 `save_silent_video` 把整段 buffer
    搬 GPU 导致保存 OOM。
  - 缓存版重跑（已修复保存为 CPU + `.pt` 备份）：进行中，跑完后补 VBench
    四项分数与本节。
- **产物路径**：
  - 日志：`NAV/result/vbench/20260805_infworld_vbench_16chunk_autoreg/logs/inference.log`
  - 视频/结果：`NAV/result/vbench/20260805_infworld_vbench_16chunk_autoreg/{videos,metrics}/`（待完成）
- **根因与结论**：原脚本 OOM 根因是脚本侧每 chunk 重 encode 整个像素 buffer
  （HPMC 不覆盖此步），论文 Figure 5 的 ~45 GB 平台是 DiT+HPMC 开销、需
  80 GB H800 才有余量。缓存 latent 版把历史维持在 latent 域，47 GB 即可
  跑通 16 chunk。需在文档标注：缓存版与原脚本在历史 latent 上差一个 VAE
  decode→encode 往返（可忽略），且仅 1 场景 vs 论文 100 场景。

#### 记录 7 — NAV A/B from-scratch 3-chunk 同协议对比（2026-08-05，已完成）

- **时间**：2026-08-05
- **模型实现与训练实现**：NAV Stage One 1.0 from-scratch（DEC-019，
  ebs16、1000 effective steps、DL3DV full-episode latent、empty UMT5）。
  - A = `latent_prefix`，checkpoint `full-step-000500.pt`（A 于 step 541 停止，
    最后 ckpt@500）。
  - B = `dit_condition`，checkpoint `full-step-000500.pt` 与
    `full-step-001000.pt`（B 训练已于 step 1000 完成，`full-final.pt` 同 step 1000）。
  - 推理用 `scripts/infer_register_vbench.py`（NAV Register 路径，Extractor/Updater
    在线更新，非 InfiniteWorld HPMC）。
- **测评条件**：单 campus 3-chunk（241 帧，约 8 秒），seed 42，30 步，CFG 5.0；
  VBench 四项（MS/DD/AQ/IQ），四项算术平均。与记录 5（InfiniteWorld 3-chunk）
  同口径，便于 NAV vs InfiniteWorld 在 3-chunk 长度下直接对比。
- **测评结果**：

| 指标 | A@500 | B@500 | B@1000 |
| --- | ---: | ---: | ---: |
| Motion Smoothness | 0.9835 | 0.9800 | 0.9804 |
| Dynamic Degree | true | false | false |
| Aesthetic Quality | 0.6150 | 0.6173 | 0.6120 |
| Imaging Quality（原始 MUSIQ） | 78.05 | 79.09 | 77.69 |
| Average Score（四项算术平均） | 0.8447 | 0.5970 | 0.5923 |

- **产物路径**：
  - 视频：`NAV/result/vbench/20260805_navA_s500_vbench_3chunk_autoreg/`、`20260805_navB_s500_vbench_3chunk_autoreg/`、`20260805_navB_s1000_vbench_3chunk_autoreg/` 下 `videos/0000_seed0042_*.mp4`
  - 指标：上述三目录下 `metrics/`
  - 日志：上述三目录下 `logs/{inference,eval}.log`
- **解读与局限**：
  - 3-chunk 下 A@500 的 Dynamic Degree 仍为 true（有运动），而 B@500 与 B@1000
    均为 false（静止）。这与记录 4（2-chunk）中 B@800 DD=true、B@1000 DD=false 的
    趋势一致：B 在更长 horizon 上更易塌缩为静止帧，A 的 latent-prefix 注入在
    维持动态上更稳。
  - MS/AQ/IQ 三者量级接近（MS~0.98、AQ~0.61、IQ~78），与 2-chunk 记录 4 量级一致，
    说明 3-chunk 未引入明显技术质量退化；差异主要在 Dynamic Degree。
  - 仅 1 场景、单 seed，统计意义有限；Dynamic Degree 为布尔量，更需多样本/多场景
    才能判定 A vs B 的动态维持能力差异。后续应扩到多场景多 seed 再下结论。

#### 记录 8 — Teacher-forced（3 GT 历史 chunk → 预测 1 chunk）三模型对比（2026-08-06，已完成）

- **时间**：2026-08-06
- **范式**：`gthist`（全部真实历史，teacher-forced 诊断线，区别于记录 1–7 的
  `autoreg` 自回归线）。给 n=3 个 GT 历史 chunk，预测第 4 个 chunk（target_chunk=3），
  再与 GT 第 4 个 chunk 比对。此为 NAV 内部"训推一致"诊断，**不与外部模型跨范式比**，
  此处 InfiniteWorld 仅作同条件参照。
- **模型实现与训练实现**：
  - NAV A = `latent_prefix`，`full-step-000500.pt`（Stage One 1.0 from-scratch，ebs16，
    DL3DV，empty UMT5）。
  - NAV B = `dit_condition`，`full-step-001000.pt`（同上，B 已训完 1000 步）。
  - InfiniteWorld = 官方权重 `infinite_world_model.ckpt`（原生 HPMC）。
  - 推理用新脚本 `scripts/infer_teacher_forced_vbench.py`：NAV 走 Register
    （extract + 2×update），InfiniteWorld 走 HPMC（image_cond = 3 chunk 沿 time 维拼接）。
- **测评条件**：
  - Episode：`full_episodes_v1/dl3dv/dl3dv__03a87672...ad0d9.pt`（DL3DV 训练集 episode，
    含 ≥4 chunk）。
  - 历史 chunks：chunk 0/1/2（GT），预测 chunk 3；action 取该 episode chunk 3 对应的
    **GT pose-derived 离散动作**（非 demo `0001.json`），即训推一致的 action。
  - **文本：统一空 text（null token）**，隔离 history+action 的单步预测能力，
    排除文本通道（Stage One 未训文本）的干扰。
  - seed 42，30 步，CFG 5.0，shift 7；输出 81 帧（1 chunk）。
  - 指标：VBench 四项（MS/DD/AQ/IQ，IQ 用归一化值）+ 保真度三项（L1↓ / SSIM↑ /
    LPIPS↓，逐帧算后取均值，gen vs GT 第 4 chunk）。
- **测评结果**：

| 指标 | NAV A@500 | NAV B@1000 | InfiniteWorld |
| --- | ---: | ---: | ---: |
| Motion Smoothness | 0.9904 | 0.9918 | 0.9916 |
| Dynamic Degree | true (1.0) | true (1.0) | true (1.0) |
| Aesthetic Quality | 0.5462 | 0.5431 | 0.5383 |
| Imaging Quality（归一化） | 0.7533 | 0.7461 | 0.7508 |
| Average Score（VBench 四项） | 0.8225 | 0.8202 | 0.8202 |
| L1↓（vs GT） | 0.0874 | 0.0917 | 0.1152 |
| SSIM↑（vs GT） | 0.5412 | 0.5449 | 0.4733 |
| LPIPS↓（vs GT） | 0.3150 | 0.3290 | 0.3726 |

- **产物路径**：
  - `NAV/result/vbench/20260806_navA_s500_vbench_3chunk_gthist/`（`videos/gen.mp4`、
    `gt/gt.mp4`、`metrics/{fidelity.json, results_*eval_results.json}`、`logs/log.txt`）
  - `20260806_navB_s1000_vbench_3chunk_gthist/`、`20260806_infworld_vbench_3chunk_gthist/`
    同结构。
- **解读与局限**：
  - **VBench 四项三者几乎并列**（平均 0.820–0.823），且 DD 全部 true——在
    teacher-forced（给真实历史、单步预测）条件下，三者技术质量无显著差异，
    说明自回归线（记录 7）中 B 的 DD 塌缩来自**误差累积**而非单步生成能力不足。
  - **保真度 NAV 优于 InfiniteWorld**：A/B 的 L1 更低、SSIM 更高、LPIPS 更低；
    InfiniteWorld L1=0.1152、SSIM=0.4733、LPIPS=0.3726 明显劣于 NAV（A: 0.0874 /
    0.5412 / 0.3150）。即在"给同样真实历史、预测同一 GT chunk"的训推一致条件下，
    NAV 的单步预测更贴近 GT——这支持 NAV Register 在单步保真上不弱于、甚至优于
    HPMC 的结论。
  - A 与 B 保真度接近（A 的 L1/LPIPS 略优，B 的 SSIM 略优），单步预测上两 variant
    无显著差距。
  - **局限**：仅 1 episode、单 seed、空 text；保真度仅反映像素级贴近度，不等于
    几何/物理正确性；InfiniteWorld 在此用的是其原生 HPMC + 官方权重，未对其做
    teacher-forced 专项适配。后续应扩多 episode/多 seed，并与几何 probe 联合分析。

---

### 改名映射表（2026-08-06 统一改名）

旧目录名 → 新目录名（均位于 `NAV/result/vbench/` 下）：

| 旧名 | 新名 | 对应记录 |
| --- | --- | --- |
| `infiniteworld_single_2chunks` | `20260723_infworld_vbench_2chunk_autoreg` | 记录 2(b) |
| `nav_a_single_2chunks` | `20260725_navA_s1000_vbench_2chunk_autoreg` | 记录 3 |
| `nav_b_single_2chunks` | `20260725_navB_s1000_vbench_2chunk_autoreg` | 记录 3 |
| `nav_a_step500_single2c` | `20260804_navA_s500_vbench_2chunk_autoreg` | 记录 4 |
| `nav_b_step800_single2c` | `20260804_navB_s800_vbench_2chunk_autoreg` | 记录 4 |
| `infiniteworld_3chunks` | `20260805_infworld_vbench_3chunk_autoreg` | 记录 5 |
| `infiniteworld_16chunks` | `20260805_infworld_vbench_16chunk_autoreg` | 记录 6 |
| `nav_a_step500_3chunks` | `20260805_navA_s500_vbench_3chunk_autoreg` | 记录 7 |
| `nav_b_step500_3chunks` | `20260805_navB_s500_vbench_3chunk_autoreg` | 记录 7 |
| `nav_b_step1000_3chunks` | `20260805_navB_s1000_vbench_3chunk_autoreg` | 记录 7 |
| `nav_a_stats10` | `20260725_navA_s1000_vbench_2chunk_autoreg_stats10` | 统计子集（2 场景×5 种子） |
| `nav_b_stats10` | `20260725_navB_s1000_vbench_2chunk_autoreg_stats10` | 统计子集 |
| `infiniteworld_stats10` | `20260725_infworld_vbench_2chunk_autoreg_stats10` | 统计子集 |
| `nav_a_stats20_gpu0` | `20260727_navA_s1000_vbench_2chunk_autoreg_stats20` | 统计子集（2 场景×10 种子） |
| `nav_b_stats20_gpu0` | `20260727_navB_s1000_vbench_2chunk_autoreg_stats20` | 统计子集 |
| `infiniteworld_stats20_gpu0` | `20260727_infworld_vbench_2chunk_autoreg_stats20` | 统计子集 |
| `threeway_stats20_gpu0` | `20260727_threeway_vbench_2chunk_autoreg_stats20` | 三模型对比汇总 |

**豁免保留原名**（无法映射到单一名称，不参与改名）：

| 目录 | 原因 |
| --- | --- |
| `infiniteworld_prior_runs` | 多条早期小复现的合集（1-chunk 6 指标、1-chunk 论文协议、2-step、30-step、recheck 等），非单次测评 |
| `frame_analysis` | 跨模型（A/B/InfWorld）的帧级 contact 分析，非 VBench 测评运行 |


## R2R-CE / RxR-CE SOTA 复现协议（≥ StreamVLN）


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-002` |
| 类型 | 评测协议与复现规范（Evaluation Protocol） |
| 状态 | StreamVLN 20-ep 闭环已完成；100/500-ep 子集已完成；全量 val_unseen Pending |
| 更新时间 | 2026-09-02 21:28（Asia/Shanghai） |
| 职责 | 规定 NAV 对 R2R-CE/RxR-CE 上分数不低于 StreamVLN 的可开源复现工作的唯一环境、数据、命令与结果落盘口径；并记录已完成子集复现的成功率与 latency |

### 当前结论

1. **StreamVLN / DualVLN 环境与官方评测入口已跑通**；权重与数据软链就绪。
2. **子集复现已完成**（均为 R2R-CE `val_unseen` **数据集顺序前 N 条**，**非全量 1839**）：
   - StreamVLN：**100-ep**、**500-ep**
   - DualVLN：**100-ep**（`diffusers==0.31.0`，本地 GPU 整模推理）
3. DualVLN 前 100 已贴近论文全量量级；StreamVLN 前 500 仍低于论文全量约 12 SR 点——子集偏差 + 未跑全量，不宜直接判装坏。
4. **推理均在本体（本地 GPU）**：`from_pretrained` 加载整模，无远端 API。DualVLN 的 7B（S2）与小 DiT（S1）同进程同卡。
5. 2026-09-02 按 StreamVLN 官方 Habitat 入口完成了 20 条 R2R `val_unseen` **闭环**复现；模型动作逐步执行并反馈下一帧，不是固定轨迹回放。
6. 环境形态与复现门槛见下文；全量 Val-Unseen 仍未跑。

本轮纳入复现队列：

| 优先级 | 方法 | 传感器 | 文献 R2R Val-Unseen SR/SPL | 仓库 | 权重 | 本地环境 |
| --- | --- | --- | ---: | --- | --- | --- |
| P0 | StreamVLN | RGB | **56.4 / 50.2** | `3d_wm_vln/StreamVLN` @`e48f6ff` | `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `virtual_env/.venv_streamvln` |
| P1 | InternVLA-N1 DualVLN | RGB | **64.3 / 58.5** | `3d_wm_vln/InternNav` @`7a5c624` | `InternRobotics/InternVLA-N1-DualVLN` | `virtual_env/.venv_internnav` |
| P2 | InternVLA-N1 Dual + NavDP* | RGB-D | **64.1 / 58.1**（或旧表 58.2/54.0） | 同上 | `InternRobotics/InternVLA-N1-w-NavDP` | 同上 |

明确不复现、只引用：Qwen-RobotNav、Robostral Navigate、以及 panoramic-only 行（见 `../04_evaluation/evaluation_reproduction_and_benchmarks.md`）。

工程约束：第三方仓在 `3d_wm_vln/<RepoName>`；环境在 `virtual_env/.venv_<name>`（conda prefix）；日志 `NAV/log/vln_ce/`；结果 `NAV/result/vln_ce/<method>/`。

---

### 复现成功率（Reproduced by us）

口径：Habitat VLN-CE，R2R `val_unseen`，成功距离 3 m；动作物理 **0.25 m / 15°**（以各仓库 yaml/代码为准）。来源标记：`Reproduced by us (partial)`。

#### StreamVLN

| Run | N | SR↑ | SPL↑ | OS↑ | NE↓ | 产物 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| closedloop_r2r20_20260902 | 20 | **55.0** | **45.03** | **70.0** | **4.21** | `result/vln_ce/streamvln/smoke20_closedloop_r2r20_20260902_211519/` |
| smoke100 | 100 | **41.0** | **36.0** | **54.0** | **6.53** | `result/vln_ce/streamvln/smoke100_20260807_220243/` |
| smoke500 | 500 | **44.2** | **37.4** | **58.0** | **6.07** | `result/vln_ce/streamvln/smoke500_20260808_021908/` |
| Paper v1_3（全量） | 1839 | 56.4 | 50.2 | 63.6 | 4.90 | Official reported |

#### StreamVLN：20 条闭环复现（2026-09-02）

- 官方入口：`StreamVLN/streamvln/streamvln_eval.py`，通过
  `NAV/scripts/eval_baselines/run_streamvln_eval.sh` 启动；`MAX_EPISODES=20`，
  `VLN_SPLIT=val_unseen`，`CUDA_VISIBLE_DEVICES=1`，`NPROC=1`。
- 数据：官方 Habitat R2R-CE 数据 `/sharedata/datasets/R2R/R2R_VLNCE_v1-3`，
  MP3D 场景 `/sharedata/datasets/mp3d/v1/tasks/mp3d`；每个 episode 中模型输出的
  `TURN_LEFT/RIGHT`、`MOVE_FORWARD`、`STOP` 会实际调用 Habitat action，直到
  episode over 或达到最大步数。
- 结果（20 条、按 episode 宏平均）：`SR=55.0%`（11/20）、`SPL=45.03%`、
  `OS=70.0%`、`NE=4.205m`。这是小样本复现，不能替代论文的 1839 条全量结果；
  与论文 `SR=56.4/SPL=50.2` 在误差范围内接近，但不宣称达到或超过官方全量。
- 产物：
  `result/vln_ce/streamvln/smoke20_closedloop_r2r20_20260902_211519/`；
  episode 明细在 `habitat_out/result.json`，官方运行日志在
  `logs/eval_smoke20_closedloop_r2r20_20260902_211519.log`。

- 官方入口：`StreamVLN/streamvln/streamvln_eval.py`；封装 `MAX_EPISODES=N bash NAV/scripts/eval_baselines/run_streamvln_eval.sh`。
- 采样：数据集顺序前 N；100-ep 约覆盖 7 个 scene。
- 对照 JSON：各 run 下 `metrics/compare_paper.json`。
- 栈：Python 3.9.23 / torch 2.1.2+cu121 / habitat_sim 0.2.4 / `flash_attn==2.5.8`。

#### DualVLN（InternVLA-N1 Dual System）

| Run | N | SR↑ | SPL↑ | OS↑ | NE↓ | 产物 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke100 | 100 | **67.0** | **57.4** | **73.0** | **4.33** | `result/vln_ce/dualvln/smoke100_20260808_082720/` |
| Paper（全量） | 1839 | 64.3 | 58.5 | 70.7 | 4.05 | Official reported |

- 官方入口：`InternNav/scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py`；封装 `run_dualvln_eval.sh`。
- 权重：`/sharedata/NAV/baselines/checkpoints/InternVLA-N1-DualVLN`（`system1=nextdit_async`）。
- **关键依赖**：`diffusers==0.31.0`（InternNav#322；新版 Lumina FFN 维与 ckpt 不匹配）。
- 栈：Python 3.9.23 / torch 2.5.1+cu121 / habitat_sim 0.2.4。
- 子集 SR 略高于论文全量属正常波动，**不能**直接宣称超过 SOTA。

#### 并排摘要

| 方法 | 子集 | Ours SR/SPL | Paper SR/SPL | ΔSR |
| --- | ---: | ---: | ---: | ---: |
| StreamVLN | 500 | 44.2 / 37.4 | 56.4 / 50.2 | −12.2 |
| DualVLN | 100 | 67.0 / 57.4 | 64.3 / 58.5 | +2.7 |

---

### Latency 体系与实测

机器可读明细：各 run 的 `metrics/latency.json`。下列为日志时间戳解析（无 CUDA event），含环境步进开销。

#### StreamVLN：单 Video-LLM 半闭环

设计（非双系统）：

| 组件 | 行为 |
| --- | --- |
| 模型 | 单一 Video-LLM；一次 `generate` 出离散动作序列 |
| `num_future_steps=4` | 约每 4 个 Habitat 步查一次模型 |
| `num_frames=32` | 滑动窗口；到点清空 KV，注入慢 memory（`num_history=8`） |
| KV cache | 窗口内 `past_key_values` 复用（论文 SlowFast 的「快」路径） |

实测（smoke100；smoke500 量级一致）：

| 指标 | 数值 |
| --- | ---: |
| 墙钟 | ~**46 s/ep**（100-ep ≈ 1.3 h） |
| Habitat 步进 | **~2.1 Hz** |
| VLM 调用 | **~0.53 Hz**（约每 **3.95** 步一次） |
| 动作块长度 | ~97% 为 **4** |
| 单次 generate（粗估） | 冷/窗口重置中位 **~1.1 s**；稳态约 **~1.2 s**（扣 4 步环境后） |

500-ep：步进 ~2.2 Hz，VLM ~0.56 Hz，墙钟约 6.0 h。

#### DualVLN：S2（7B VLM）+ S1（小 DiT）

设计：

| 系统 | 是什么 | 规模 | 调度（Habitat CE） |
| --- | --- | ---: | --- |
| **S2** | Qwen2.5-VL-7B：像素目标 / 视角调整 + latent plan | ~8.3B | 像素 plan 约每 **8** 步刷新；视角调整更勤 |
| **S1** | NextDiT async + DepthAnything-ViT-S；复用 `traj_latents` | ~91M | 像素目标下每 **4** 步重规划（`MAX_LOCAL_STEPS`） |

论文真机宣称 S2≈2 Hz / S1≈30 Hz（含 TensorRT）；**Habitat 复现达不到该频率**。

实测（smoke100，`CUDA_VISIBLE_DEVICES=1`，本体整模）：

| 指标 | 数值 |
| --- | ---: |
| 墙钟 | ~**85 s/ep**（100-ep ≈ 2.4 h） |
| Habitat 步进 | **~1.05 Hz** |
| S2 调用 | **~0.27 Hz**（约每 3.9 步；像素 874 / 视角离散 1395） |
| S1 调用 | **~0.29 Hz**（首推 874 + 重规划 1586） |
| S2 墙钟（含 look-down/up） | 视角/离散中位 **~1.9–2.2 s**；像素中位 **~2.7–2.8 s** |
| S2 纯 generate（粗估） | 视角/离散 **~1.5 s**；像素 **~2.3 s** |
| S1 首次（latents+traj） | 中位 **~2.6 s** |
| S1 重规划（纯小模型路径） | 中位 **~0.35 s** |

说明：

- S2 **在本地 GPU 推理**，不是云端。
- 每决策步强制 look-down×2 + look-up×2（约 **0.4–0.5 s**，且不计 `step_id`）。
- S1 快是因为不是 VLM；重规划不重跑 7B。

#### 对比（同机子集复现）

| | StreamVLN | DualVLN |
| --- | ---: | ---: |
| 架构 | 单 VLM | S2 7B + S1 ~91M |
| 步进墙钟 | ~2.1 Hz | ~1.05 Hz |
| 大模型调用 | ~0.53 Hz | S2 ~0.27 Hz |
| 快路径 | 窗口内 KV 续写 | S1 DiT ~0.35 s/次 |
| 典型 ep 墙钟 | ~46 s | ~85 s |

---

### 方法或依据

#### 门槛与分组规则

- 主比较轴：**R2R-CE Val-Unseen SR**；RxR-CE 同步报告 SR/SPL/nDTW。
- 同一表格内必须标注传感器（RGB / RGB-D / panoramic），禁止跨传感器宣称“超过”。
- 来源列必填：`Official reported` / `Reproduced by us` /
  `Re-evaluated from released checkpoint`（见 NAV-RES-002）。
- 子集结果必须标注 N 与采样方式；不得与全量 Official 数字混称为已复现 SOTA。

#### 路径约定

| 用途 | 路径 |
| --- | --- |
| 仓库根 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln` |
| StreamVLN 代码 | `<根>/StreamVLN` |
| InternNav 代码 | `<根>/InternNav` |
| StreamVLN venv | `<根>/virtual_env/.venv_streamvln`（conda prefix，Python **3.9.23**） |
| InternNav venv | `<根>/virtual_env/.venv_internnav`（conda prefix，Python **3.9.23**） |
| 机器配置 | `NAV/config/r2r_ce_sota_reproduction.env` |
| 执行脚本 | `NAV/scripts/eval_baselines/` |
| 公共数据 | `/sharedata/datasets/{R2R,RxR,mp3d,ScaleVLN}` |
| 权重缓存 | `/sharedata/NAV/baselines/checkpoints/` |
| 评测产物 | `NAV/result/vln_ce/<method>/<run_id>/` |
| 运行日志 | `NAV/log/vln_ce/<method>/` |

#### Habitat / 二进制依赖说明

`habitat-sim==0.2.4` 的 conda 构建仅支持 **Python 3.9**，且不能注入纯
`python -m venv` 目录。因此 `.venv_*` 使用：

```bash
conda create -y -p virtual_env/.venv_streamvln python=3.9 pip
conda install -y -p virtual_env/.venv_streamvln habitat-sim=0.2.4 withbullet headless \
  -c conda-forge -c aihabitat
```

其余 PyTorch / 项目依赖用该前缀内的 `pip` 安装。`setup_*.sh` 已按此实现。
InternNav 侧须钉死 **`diffusers==0.31.0`**（见 `setup_internnav_env.sh`）。

#### 数据链接（只读软链，不复制）

由 `scripts/eval_baselines/link_vln_ce_data.sh` 创建：

```text
StreamVLN/data/datasets/r2r      → /sharedata/datasets/R2R/R2R_VLNCE_v1-3
StreamVLN/data/datasets/rxr      → /sharedata/datasets/RxR/raw/rxr_ce/RxR_VLNCE_v0
StreamVLN/data/scene_datasets/mp3d → /sharedata/datasets/mp3d/v1/tasks/mp3d
InternNav 侧同样链到上述公共目录（脚本内按官方期望布局调整）
```

InternNav Habitat 场景根应指向 `.../tasks`（使 `mp3d_ce/<scene>` 解析正确），避免双重 `mp3d/`。

#### 标准评测命令

```bash
bash NAV/scripts/eval_baselines/link_vln_ce_data.sh
bash NAV/scripts/eval_baselines/setup_streamvln_env.sh
bash NAV/scripts/eval_baselines/setup_internnav_env.sh
bash NAV/scripts/eval_baselines/download_checkpoints.sh

MAX_EPISODES=100 CUDA_VISIBLE_DEVICES=0 bash NAV/scripts/eval_baselines/run_streamvln_eval.sh
MAX_EPISODES=500 CUDA_VISIBLE_DEVICES=1 bash NAV/scripts/eval_baselines/run_streamvln_eval.sh
MAX_EPISODES=100 CUDA_VISIBLE_DEVICES=1 bash NAV/scripts/eval_baselines/run_dualvln_eval.sh
```

#### 每次 run 必须记录的字段

写入 `NAV/result/vln_ce/<method>/<run_id>/metrics/`：

- `summary.json` / `compare_paper.json` /（若测过）`latency.json`
- git commit（第三方仓库 + NAV）；venv 与版本；split、N、采样方式
- 传感器、动作步长（本复现为 **0.25 m / 15°**）；checkpoint
- SR / SPL / NE / OS；墙钟；GPU；来源标记 `Reproduced by us`

### 执行状态快照（2026-08-08）

| 步骤 | 状态 | 说明 |
| --- | --- | --- |
| clone / 数据软链 / 权重 | 完成 | StreamVLN + DualVLN ckpt |
| `.venv_streamvln` / `.venv_internnav` | 完成 | 见上文栈 |
| StreamVLN 1-ep / 100-ep / 500-ep | **完成** | 见成功率表 |
| DualVLN 100-ep | **完成** | `diffusers==0.31.0`；GPU1 |
| StreamVLN / DualVLN 全量 val_unseen | 未跑 | `MAX_EPISODES=-1` |
| DualVLN + NavDP* | 未跑 | P2 |
| RxR-CE | 未跑 | — |

### 限制与未决问题

1. `.venv_*` 实际是 conda prefix（非纯 venv），因 habitat-sim 0.2.4 仅 py3.9 conda 包。
2. 全量 Val-Unseen 未跑；子集数字不可替代 Official 全量行。
3. Latency 由日志时间戳估算，非 CUDA profiler；含 Habitat / look-down 开销。
4. DualVLN 官方 req 中的 diffusers 版本与可加载版本不一致，以 **0.31.0** 为准。
5. Qwen-RobotNav / Robostral / panoramic 行不进复现队列。

### 相关资源

- Benchmark 数字与引用链：`../04_evaluation/evaluation_reproduction_and_benchmarks.md`（NAV-RES-002）
- VLN 数据状态：`../02_data/data_preparation_schema_and_status.md`（NAV-DAT-007）
- 资源总账：`../06_operations/resource_inventory.md`
- 决策：`../00_overview/decision_log.md` DEC-021
- 配置：`../../config/r2r_ce_sota_reproduction.env`
- 脚本：`../../scripts/eval_baselines/`
- StreamVLN 结果：`../../result/vln_ce/streamvln/smoke{100_20260807_220243,500_20260808_021908}/`
- DualVLN 结果：`../../result/vln_ce/dualvln/smoke100_20260808_082720/`


## V1 BridgeVLA++ memoryBench 复现记录


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-003` |
| 类型 | 外部基线复现 |
| 状态 | Smoke Reproduced |
| 更新时间 | 2026-08-08 |
| 职责 | 记录 BridgeVLA++ memory-enhanced 版本在 memoryBench 上的本地复现命令、结果和限制 |

### 当前结论

BridgeVLA++ memory-enhanced 版本已经跑通 memoryBench 最小复现。使用官方
`model_160.pth`、官方 server/client、`TEMPORAL_MEMORY=true`、`SPATIAL_MEMORY=true`。

本次稳定 smoke 使用：

- task filter：`put_block_back`
- taskvars：`put_block_back+0/+1/+2/+3`
- `NUM_EPISODES=1`
- `MAX_EPISODES=1`
- `MAX_STEPS=25`
- `VISUALIZE=0`
- `RECORD_VIDEO=0`

结果：4 条 rollout 全部 success，平均 success rate = 100%。这不是完整 leaderboard
指标，只证明官方 checkpoint、memory 配置、RLBench/CoppeliaSim 和 server/client 链路在
本机可用。

### 结果路径

| 结果 | 路径 | 说明 |
| --- | --- | --- |
| 3-step 链路 smoke | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/bridgevla_plus_memorybench/smoke/model_160/seed608/result.jsonl` | 4/4 rollout 正常结束，但 `MAX_STEPS=3`，全部未完成任务 |
| 25-step 有效 smoke | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/bridgevla_plus_memorybench/smoke_25step/model_160/seed608/result.jsonl` | 4/4 success，实际每条 `nsteps=12` |

25-step result 摘要：

| task filter | taskvars 数 | episodes/taskvar | max steps | success |
| --- | ---: | ---: | ---: | ---: |
| `put_block_back` | 4 | 1 | 25 | 4 / 4 |

### 复现命令

先启动 server：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA

CONDA_BASE=/sharedata/anaconda3 \
GEMBENCH_CONDA_ENV=bridgevla_plus_gembench \
BRIDGEVLA_DATA_ROOT=/sharedata/BridgeVLA/data/bridgevla_data \
BRIDGEVLA_CKPT_ROOT=/sharedata/BridgeVLA/data/bridgevla_ckpt \
HF_HOME=/sharedata/BridgeVLA/cache/hf \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
PORT=13169 \
bash finetune/memoryBench/run_server.sh \
  160 /sharedata/BridgeVLA/data/bridgevla_ckpt/bridgevla_plus/memorybench
```

server 输出 `Agent ready.` 和 `Running on http://localhost:13169` 后，另开 client：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA

CONDA_BASE=/sharedata/anaconda3 \
GEMBENCH_CONDA_ENV=bridgevla_plus_gembench \
BRIDGEVLA_DATA_ROOT=/sharedata/BridgeVLA/data/bridgevla_data \
BRIDGEVLA_CKPT_ROOT=/sharedata/BridgeVLA/data/bridgevla_ckpt \
HF_HOME=/sharedata/BridgeVLA/cache/hf \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
IP=localhost \
PORT=13169 \
TASKS=put_block_back \
NUM_EPISODES=1 \
MAX_EPISODES=1 \
MAX_STEPS=25 \
VISUALIZE=0 \
RECORD_VIDEO=0 \
OUTPUT_ROOT=/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/bridgevla_plus_memorybench/smoke_25step \
bash finetune/memoryBench/run_client.sh \
  /sharedata/BridgeVLA/data/bridgevla_ckpt/bridgevla_plus/memorybench 160
```

### 运行观测

- server 首次加载慢，主要在 PaliGemma 与 `model_160.pth` 初始化；二次加载因系统 cache
  明显更快。
- 25-step smoke 中每个 taskvar 用时约 12 秒；server 端每个动作 step 对应一次
  `POST /predict`。
- GPU0 在 server 加载后显存约 9.7GB；client 主要负责仿真 rollout。
- client 退出时出现 `QMutex: destroying locked mutex`，但进程返回码为 0，结果文件正常写入。

### 与官方完整指标的关系

官方 BridgeVLA++ 页面与 README 报告 memoryBench 上 BridgeVLA++ 约 99.7 success rate，
原 BridgeVLA 约 11.3。本文档的 4-episode smoke 不能替代官方完整评测；如需正式复现，
应补齐并展开全部 memoryBench test task，使用官方默认：

- all taskvars；
- `NUM_EPISODES=25`；
- `MAX_STEPS=25`；
- 固定 seed；
- 输出到 `NAV/result/bridgevla_plus_memorybench/full_eval/<run_id>/`。

### 已知限制

1. 本次为了快速确认 memory-enhanced 版本可用，只展开并评测了 `test/put_block_back`。
2. `train/put_block_back` 曾在官方解压脚本执行中被中断后保留了部分已展开目录；zip 原包保留，
   后续训练或完整审计前应重新运行官方解压脚本补齐。
3. `hf-mirror` 对 `hqfang/memorybench` 的 metadata 请求不兼容；缺失 zip 最终使用
   Hugging Face resolve 直链 `wget -c` 下载完成。


## V1 Fast-WAM 与 GigaWorld-Policy 复现记录


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-004` |
| 类型 | 外部 WAM 基线复现 |
| 状态 | Smoke Reproduced |
| 更新时间 | 2026-08-09 |
| 职责 | 记录 Fast-WAM 与 GigaWorld-Policy-0.5 的本地仓库、环境、权重、最小 smoke、latency 复现和阻塞 |

### 当前结论

本轮复现选择 `Fast-WAM` 与 `GigaWorld-Policy-0.5`，目标不是先跑完整机器人
leaderboard，而是先复现二者对 Wan/WAM backbone 的使用方式：

- Fast-WAM：验证 release checkpoint 的 `infer_action` 路径，即先用当前图像
  经过 video expert 得到 first-frame latent/video cache，再用 action expert 做多步
  action denoising；推理时不生成 dense future video。
- GigaWorld-Policy-0.5：优先验证 `CasualWorldActionTransformer_MoT` 的
  `action_only=True` / prefix-cache 路径；完整 open-loop server/client 需要额外
  Wan Diffusers base model 与 LeRobot v3 数据。

截至 2026-08-09 03:11，两个官方仓库已 clone，虚拟环境已配置，权重已落盘，
NAV 侧最小 smoke 已跑通。GigaWorld 官方 `requirements.txt` 存在
`diffusers==0.36.0` 与 `lerobot==0.4.4` 的 dependency conflict，本地采用
“安装除 `lerobot` 外的官方依赖，再 `--no-deps` 安装 `lerobot==0.4.4`”的方式保持
`diffusers==0.36.0`，不修改官方代码。

### Smoke 结果

| 项 | Fast-WAM | GigaWorld-Policy-0.5 |
| --- | ---: | ---: |
| 运行时间 | 2026-08-09 03:10 | 2026-08-09 02:56 |
| GPU | RTX 6000 Ada | RTX 6000 Ada |
| checkpoint | `libero_uncond_2cam224.pt` | `Giga-World-Policy-0.5` |
| denoising steps | 2 | 2 |
| load seconds | 63.19 | 151.45 |
| infer seconds | 4.35 | 4.11 |
| seconds / denoise step | 2.18 | 2.05 |
| peak allocated GiB | 14.44 | 11.34 |
| 输出 shape | action `[32,7]` | action `[1,48,16]` |
| 结果 JSON | `NAV/result/fastwam/smoke/20260809_031029/summary.json` | `NAV/result/gigaworld_policy/smoke_transformer/20260809_025612/summary.json` |

GigaWorld 的 `infer_seconds=4.11s` 包含首次 forward 的 PyTorch Inductor warmup /
compile 开销。两步分别为 `4.07s` 与 `0.036s`，因此后续正式 latency 统计需要区分
首步 warmup 和 steady-state。

### 路径

| 项 | Fast-WAM | GigaWorld-Policy |
| --- | --- | --- |
| 代码仓库 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy` |
| commit | `45d8e14` | `f3d5a88` |
| 虚拟环境 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_fastwam` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_gigaworld_policy` |
| 公共权重/数据 | `/sharedata/FastWAM` | `/sharedata/GigaWorld-Policy` |
| log | `NAV/log/fastwam` | `NAV/log/gigaworld_policy` |
| result | `NAV/result/fastwam` | `NAV/result/gigaworld_policy` |

### 执行入口

Fast-WAM transformer/action-only smoke：

```bash
GPU_ID=0 STEPS=2 bash /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/reproduce_wam/run_fastwam_smoke.sh
```

GigaWorld-Policy-0.5 transformer-only smoke：

```bash
GPU_ID=0 STEPS=2 bash /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/reproduce_wam/run_gigaworld_transformer_smoke.sh
```

两个脚本都只封装 NAV 侧路径和最小随机输入，不修改官方 repo。输出为：

```text
NAV/result/fastwam/smoke/<run_id>/summary.json
NAV/result/gigaworld_policy/smoke_transformer/<run_id>/summary.json
```

### 关键日志

| 任务 | 日志或结果 |
| --- | --- |
| Fast-WAM 安装 | `NAV/log/fastwam/install_20260809_023904.log` |
| Fast-WAM ActionDiT 预处理与 smoke | 前台复现 run id `20260809_031029`；结果见 `NAV/result/fastwam/smoke/20260809_031029/summary.json` |
| GigaWorld 安装修复 | `NAV/log/gigaworld_policy/install_repair_20260809_024618.log` |
| GigaWorld transformer smoke | `NAV/result/gigaworld_policy/smoke_transformer/20260809_025612/summary.json` |

### 复现口径

#### Fast-WAM

使用官方 `libero_uncond_2cam224.pt` 和 `libero_uncond_2cam224_dataset_stats.json`。
若 `/sharedata/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
不存在，脚本会先按官方 `scripts/preprocess_action_dit_backbone.py` 从 Wan2.2 预处理
ActionDiT backbone。本机需要显式关闭 `redirect_common_files`，否则 DiffSynth 会尝试
访问不可用的 `DiffSynth-Studio/Wan-Series-Converted-Safetensors`；NAV 脚本会临时生成
local-only config，只使用 `/sharedata/FastWAM/models` 下的本地 Wan 权重。

最小 smoke 使用：

- random image：`[1,3,224,448]`，范围 `[-1,1]`；
- random/zero context embedding：`[1,128,4096]`，绕开 T5 text encoder；
- proprio：按 `libero_2cam` 配置使用 8 维 zero proprio；
- action horizon：默认 32；
- denoising steps：默认 2，用于快速验证链路，正式 latency 可改为论文/官方默认 steps。

该路径验证的是 Fast-WAM 的 `infer_action`：

```text
image -> Wan VAE first-frame latent
      -> video expert pre_dit
      -> MoT prefill_video_cache
      -> action expert 多步 denoise
      -> action sequence
```

#### GigaWorld-Policy-0.5

GigaWorld 完整 open-loop server/client 需要：

- transformer checkpoint：`/sharedata/GigaWorld-Policy/Giga-World-Policy-0.5`；
- Wan Diffusers base：`/sharedata/GigaWorld-Policy/base_models/Wan2.2-TI2V-5B-Diffusers`；
- norm stats JSON；
- LeRobot v3 packed dataset root 或等效 dummy/open-loop episode。

当前优先 smoke 是 transformer-only，直接加载官方 MoT transformer checkpoint，构造：

- `ref_latents=[1,48,1,24,20]`；
- `action=[1,48,16]`；
- `state=[1,1,16]`；
- `encoder_hidden_states=[1,64,4096]`；
- `action_only=True`。

该路径验证的是 GigaWorld 的 cached action-only forward：

```text
ref_latents + state -> prefix cache
action noise + timestep -> forward_action_stack_with_prefix_cache
                         -> action denoise prediction
```

### 已知问题

1. GigaWorld 官方依赖 pin 互相冲突：`diffusers==0.36.0` 与 `lerobot==0.4.4`
   的 metadata 要求不一致。本地修复策略只影响安装解析，不改模型代码。
2. GigaWorld 完整 open-loop 需要 Wan Diffusers layout；本机已有
   `/sharedata/Wan2.2-TI2V-5B` 是原生 Wan layout，不能直接作为 `BASE_MODEL`。
3. Fast-WAM 完整 LIBERO/RoboTwin benchmark 仍需官方仿真环境与 benchmark assets；
   当前先跑不依赖仿真的 model smoke。
4. 早先一次 Fast-WAM `libero_uncond_2cam224.pt` 由 aria2 产生了尾部全 0 的坏文件，
   虽然文件大小等于 metadata，但 `torch.load` 报
   `failed finding central directory`。已保留为
   `/sharedata/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt.bad_20260809_0305`，
   并用 `hf_hub_download('yuanty/fastwam', 'libero_uncond_2cam224.pt')` 重下。

### 下一步验收

1. 若需要正式 latency，分别记录 warmup 后的 steady-state 单步时间、官方默认 steps
   和完整 action chunk 时间。
2. 若 GigaWorld transformer-only smoke 通过，再补完整 open-loop server/client
   dummy episode 或小型真实 LeRobot episode。
3. 若需要 robot benchmark 分数，再配置 LIBERO/RoboTwin 或 GigaWorld 官方
   LeRobot v3 评测数据；当前 smoke 只验证 backbone/action-only 前向链路。


## R2R / RxR Benchmark 与最新 SOTA 调研


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-002` |
| 类型 | Benchmark / Leaderboard 调研 |
| 状态 | Verified Snapshot / External Results |
| 更新时间 | 2026-08-07 |
| 职责 | 统一 R2R、R2R-CE、RxR、RxR-CE 的评测口径，核对论文引用方式，并给出 NAV 使用的最终成功率基准 |

### 本地复现队列（2026-08-07）

NAV 实际 Habitat 复现只覆盖 **R2R Val-Unseen SR ≥ StreamVLN** 且有公开权重的方法（含
StreamVLN）。工程约束与命令见 **`../04_evaluation/evaluation_reproduction_and_benchmarks.md`
（NAV-EVL-002，DEC-021）**：仓库在 `3d_wm_vln/{StreamVLN,InternNav}`，venv 在
`virtual_env/.venv_{streamvln,internnav}`。Qwen-RobotNav / Robostral 等更高分但无权重
的工作仍只作本文件的引用追踪，不进入复现队列。

### 当前结论

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

### Benchmark 边界

| 名称 | 环境与动作 | 语言 | 常用 split | 首要指标 |
| --- | --- | --- | --- | --- |
| R2R | Matterport3D navigation graph，节点间跳转 | English | Val-Unseen / Test-Unseen | SR、SPL |
| R2R-CE | Habitat 连续空间，低层离散动作 | English | Val-Unseen / Test-Unseen | SR、SPL |
| RxR | Matterport3D navigation graph | English/Hindi/Telugu | Val-Unseen / Test-Standard | NDTW、SDTW、SR、SPL |
| RxR-CE / RxR-Habitat | Habitat 连续空间，低层离散动作 | English/Hindi/Telugu | Val-Unseen / Test-Challenge | **NDTW**、SDTW、SR、SPL |

官方 RxR 说明中，RxR 使用 Test-Standard，而 RxR-Habitat 使用 Test-Challenge；两者
不是同一个测试集。RxR-Habitat 标准配置为30度转向/俯仰、0.25m前进和
480×640 RGB-D。RxR-Habitat leaderboard 按 NDTW 排名，而不是只按 SR 排名。

#### Success Rate 定义

R2R-CE/RxR-CE 中通常在 agent 主动 `STOP` 后，若最终位置距目标不超过3m，则该
episode 成功。SR 是成功 episode 比例：

\[
\mathrm{SR}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[d_i\leq3\mathrm{m}].
\]

SPL 同时惩罚绕路；NDTW 衡量整条预测路径与参考路径的对齐程度。RxR 指令更长且
强调严格路径跟随，所以只看 SR 会漏掉“到达了终点但没有按指令走”的错误。

### R2R-CE 结果梳理

#### 官方起点

VLN-CE 官方仓库给出的 CMA+PM+DA+Aug leaderboard baseline：

| Split | SR | SPL |
| --- | ---: | ---: |
| Val-Unseen | 29% | 27% |
| Test | 28% | 25% |

官方同时说明，论文曾报告 Val-Unseen SPL=30%，leaderboard 对同一模型重评为27%。
这提供了直接证据：不同 Habitat/硬件构建和评测入口会令论文数字与榜单数字不一致。

来源：[VLN-CE 官方仓库](https://github.com/jacobkrantz/VLN-CE#vln-ce-challenge-r2r-data)。

#### 代表性连续环境进展

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

#### 为什么不把81.1%直接写成 R2R-CE 最终 SOTA

StereoNav 摘要给出 R2R-CE SR/SPL=81.1/68.3，但摘要没有明确说明是 Val-Unseen
还是 Test、是否使用标准 RGB-D/panoramic 配置、是否采用额外视觉先验以及是否提交
官方 leaderboard。它可以记录为“最新 self-reported claim”，但当前证据不足以替换
60/52这一可追溯的 peer-reviewed Test 数字。

同理，HSAN 的64/59是 Val-Unseen，不应与 NavMorph 的 Test 60/52直接比较高低。

### RxR-CE 结果梳理

#### 官方挑战赛确定值

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

#### 后续论文结果

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

### Qwen-RobotNav 专项追踪

#### 模型、训练和开放状态

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

#### 官方 Table 1：必须按传感器分组

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

#### 引用与对比链

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

#### 对 NAV 的直接启示

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

### 论文是否直接引用别人的成功率

答案是：**普遍会，而且对 Test split 基本不可避免；但规范论文会通过引用、符号或
脚注明确来源。**

#### 常见做法

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

#### 对 NAV 写论文的要求

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

### NAV 最终采用的成功率表

#### 主表：连续环境、Test 口径

| Benchmark | Baseline | SR | SPL | NDTW | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| R2R-CE Test | CMA official baseline | 28 | 25 | — | 最低官方基线 |
| R2R-CE Test | NavMorph + HNR | **60** | **52** | 约57 | 当前 peer-reviewed SOTA 目标 |
| RxR-CE Test-Challenge | Reborn official winner | **45.82** | **38.82** | **55.43** | 严格 challenge 对比 |
| RxR-CE Test | NavMorph + HNR | **54.98** | **43.02** | **57.31** | 当前后续 peer-reviewed Test 目标 |

#### 辅表：Val-Unseen 研究追踪

| Benchmark | 可靠已发表高值 | 最新 preprint claim | 使用规则 |
| --- | --- | --- | --- |
| R2R-CE Val-Unseen | HSAN 64/59 SR/SPL | Qwen-RobotNav panoramic 8B 72.1/66.6；Robostral monocular 77.4 SR；StereoNav 81.1/68.3口径不清 | 按 sensor 分组，不替代 Test 主表 |
| RxR-CE Val-Unseen | HSAN 59/54 SR/SPL | Qwen-RobotNav panoramic 8B 76.5/65.7、monocular 8B 73.4/63.5；Robostral monocular 75.1 SR | 同时报告 NDTW/SDTW，按 sensor 分组 |

#### NAV 的阶段性目标

| 等级 | R2R-CE SR | RxR-CE SR | 解释 |
| --- | ---: | ---: | --- |
| 可用 baseline | 约45% | 约35% | 已超过早期 recurrent baseline |
| 强基线 | 55% | 45% | 接近 ETPNav/Reborn 水平 |
| 论文竞争线 | 60% | 55% | 接近 peer-reviewed Test 高值 |
| 最新 claim 追踪线 | 72%+ | 75%+ | 对应2026 Val-Unseen foundation-model claim；必须统一 sensor 口径 |

这些数值是对**完整 benchmark split 的 episode-level SR**，不是训练集准确率、动作
预测 accuracy，也不是少量 episode smoke test。

### 限制与未决问题

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

### 相关资源

- 本地 R2R episodes：`/sharedata/datasets/R2R`
- 本地 MP3D scenes：`/sharedata/datasets/mp3d/v1/tasks/mp3d`
- 复现协议：`../04_evaluation/evaluation_reproduction_and_benchmarks.md`
- 资源总账：`../06_operations/resource_inventory.md`
- 数据准备规范：`../02_data/data_preparation_schema_and_status.md`
- Stage Three 导航设计：`../01_model/model_evolution_and_current_architecture.md`


## 世界模型训练算力与速度调研


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-001` |
| 类型 | 外部调研（Research Note） |
| 状态 | Reference |
| 更新时间 | 2026-07-28 |
| 职责 | 汇总公开世界模型训练规模、速度和可比性限制 |

更新时间：2026-07-24。

本文只记录论文、官方仓库或官方项目页能够核实的数据。论文没有公开的项目不根据
模型规模臆测训练时长；数据视频时长也不等同于训练 wall-clock time。

### 1. Infinite-World

来源：

- 论文：https://arxiv.org/abs/2602.02393
- 代码：https://github.com/MeiGen-AI/Infinite-World

公开训练设置：

- Backbone：Wan2.1-T2V-1.3B。
- 两阶段训练均为480P，AdamW，恒定 learning rate `1e-5`。
- Pre-training：30小时以上互联网第一视角与探索视频；直接压缩 history，最多4个
  temporal chunks。
- RDD fine-tuning：约30分钟 revisit-dense 实拍视频；历史上下文从1张图扩展至
  16 chunks，并启用 hierarchical compression。
- HPMC：3D-ResNet，时间压缩率4；5个重叠窗口，每窗64帧；最多320帧历史被压成
  固定 `Tmax=20`。
- 硬件：所有实验使用16张 NVIDIA H800。

没有公开：

- 两阶段的 steps、global batch size、每步耗时、总训练小时数或GPU-hours。
- 官方仓库没有发布完整训练入口与 optimizer loop，主要是推理实现。因此无法从
  公开信息可靠推回训练速度。

### 2. FantasyWorld

来源：

- 论文：https://arxiv.org/abs/2509.21657
- 代码：https://github.com/Fantasy-AMAP/fantasy-world

公开训练设置：

- Backbone：冻结的 Wan2.1-I2V-14B；训练约18万视频 clips。
- 数据包括 RealEstate10K、ACID、DL3DV、WildRGB、ScanNet、TartanAir。
- AdamW，learning rate `1e-5`。
- Stage 1（latent bridging）：只训练 geometry branch，20,000 steps，
  global batch 64；64张 H20，36小时。
- Stage 2（unified co-optimization）：81帧，`592×336` 或 `336×592`；
  只训练双向 cross-attention 与 camera-control adapter，核心 video/geometry
  backbones 冻结；10,000 steps，global batch 112；112张 H20，144小时。

由官方数字直接换算：

| 阶段 | 秒/optimizer step | Aggregate clips/s | GPU-hours |
| --- | ---: | ---: | ---: |
| Stage 1 | 6.48 | 9.88 | 2,304 H20·h |
| Stage 2 | 51.84 | 2.16 | 16,128 H20·h |
| 合计 | - | - | 18,432 H20·h |

这里的step速度是整个64/112卡集群完成一个global step的速度，不能与单卡step
直接比较。Stage 2虽然只训练adapter，仍需要让14B视频分支和geometry分支完成
81帧高分辨率前向与反向到adapter，因此每步很慢。

### 3. LingBot-World

来源：

- 论文：https://arxiv.org/abs/2601.20540
- 代码：https://github.com/Robbyant/lingbot-world

公开训练方法：

- 从 Wan2.2-I2V 初始化；两个约14B的high-noise/low-noise experts，总参数约28B，
  单个timestep只激活一个expert。
- Middle-training 采用5秒到60秒的 progressive curriculum，同时训练I2V与
  video continuation。
- Action stage冻结主DiT，只训练 action projection 与 AdaLN adapter。
- 使用 activation checkpointing、FSDP2 和 Ulysses context parallel。
- Post-training 先进行 block-causal/diffusion-forcing adaptation，再进行
  self-rollout、truncated gradient、DMD和adversarial distillation。

没有公开：

- 训练数据总量、各阶段steps、batch size、GPU型号/数量、wall-clock time和
  GPU-hours。
- 官方仓库仅发布推理代码，没有训练脚本。

论文公开的 `16 FPS @ 480P` 是 LingBot-World-Fast 的推理吞吐。原文写的是
“one GPU node”，官方命令使用 `torchrun --nproc_per_node=8`、Ulysses size 8，
因此不能将该数字写成“单张GPU 16 FPS”，更不能当作训练速度。

### 4. 公开程度更好的参照

#### SANA-WM

官方项目：https://nvlabs.github.io/Sana/WM/

- 2.6B，约21.3万公开视频 clips。
- 主训练使用64张 H100、15天，即约23,040 H100·h。
- VAE适配另需约3.5天×64 H100，即约5,376 H100·h。
- 5秒阶段为每GPU batch 1；分钟级阶段使用CP=2，相当于每GPU 0.5 clip，
  global batch 32。
- 公开训练脚本、FSDP2/CP配置和部分蒸馏训练配置，是更适合做工程速度基准的项目。

#### Endless World

论文：https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Endless_World_Real-Time_3D-Aware_Long_Video_Generation_CVPR_2026_paper.pdf

- Wan2.1-1.3B，832×480，使用4张H100训练。
- 论文没有给出总训练时长、steps或batch，因此仍不能计算训练吞吐。
- 单张H100推理约17 FPS；这是蒸馏后的推理速度，不是训练速度。

### 5. 对NAV实验的含义

现有工作支持以下判断：

1. 480P视频DiT全参训练本来就是高计算量任务。Infinite-World使用16张H800，
   并非单张48GB卡完成训练。
2. 工业项目依赖FSDP2与context parallel。LingBot-World明确把模型参数/optimizer
   state与长序列分别切到多卡；我们的当前NAV是单卡承载一个完整实验。
3. FantasyWorld选择冻结14B核心backbone，只训练geometry/adapter；即便如此，
   Stage 2在112张H20上仍需51.84秒/global step。
4. 日志中的step必须连同global batch与并行规模比较。`2秒/step`本身不能说明
   单样本或单位token更快。
5. 当前NAV的首要优化方向应是减小时空token、启用多卡context parallel/FSDP，
   其次才是常规算子优化。仅比较1.4B与3B参数量会严重误判视频模型训练成本。


## V1 Representation-only WAM 与 Video-Action Co-training 调研


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-003` |
| 类型 | 外部调研 / 复现候选选择 |
| 状态 | Active / Reproduction Targets Selected |
| 更新时间 | 2026-08-11 |
| 职责 | 调研“训练时 video/world supervision、推理时 action-only 或 compact future representation”的 WAM 路线，并确定第一批复现目标 |

### 当前结论

近期 WAM（World-Action Model）论文正在从早期的 **imagine-then-act** 转向
**representation-only / action-centered**：视频生成或未来视觉建模主要作为训练时
监督，用来塑造物理/时空表征；推理时不再显式生成 dense future video，而是直接从
compact observation / latent / action-query tokens 解码动作。这条路线与 NAV 的矛盾
完全同构：Stage One 需要 dense video generation 训练 world representation，Stage Three
需要 small input policy 做实时闭环。

第一批建议复现两个目标：

1. **Fast-WAM**：最直接回答“是否需要 test-time future imagination”。训练时保留
   video co-training，推理时跳过未来视频生成，直接 action generation。官方代码可用，
   LIBERO/RoboTwin 入口清晰。
2. **GigaWorld-Policy / GigaWorld-Policy-0.5**：action-centered WAM。训练时 future
   visual dynamics 监督 action representation，推理时 action-only decoding；其
   Mixture-of-Transformers / expert routing 对 NAV 的“video branch 与 policy branch
   共 backbone 但低延迟解耦”最有结构启发。

UVA 作为第三路线的思想先驱必须写入 related work，但复现优先级排在第二梯队：
它的“joint video-action latent + decoupled decoding”最干净，但主要是 2025 RSS 工作，
和我们当前需要的 Wan/DreamZero 时代 WAM 复现链路距离更远。

### 论文脉络

#### 1. 早期 imagine-then-act：显式视频计划驱动动作

代表包括 UniPi、Video Policy / Video Generators are Robot Policies、Video Prediction
Policy 等。核心范式是：

```text
observation + instruction
    -> video generation / future visual plan
    -> inverse dynamics or action decoder
    -> robot action
```

优点是 future video 解释性强，视频数据可作为通用训练信号；缺点是推理时必须
执行 dense future generation，延迟高，且 action accuracy 受 future video 质量牵制。
这一路线证明了“视频模型可提供 policy representation”，但不是 NAV Stage Three
可直接采用的实时路径。

#### 2. Joint WAM：video token 与 action token 在同一 DiT 中联合 denoise

代表包括 DreamZero、Unified World Model（UWM）等。核心范式是：

```text
[noisy video tokens | noisy action tokens | state tokens]
        + text / image / timestep conditions
            -> shared DiT / multimodal diffusion transformer
            -> video prediction head + action prediction head
```

DreamZero 的实现已经在本项目中复现和记录：它把 action/state register 拼入 Wan
DiT 主序列，DiT 同时输出 `video_noise_pred` 与 `action_noise_pred`。UWM 则进一步
把 video/action 作为 multimodal diffusion variables，并允许 separate diffusion timesteps，
从而统一 policy、forward dynamics、inverse dynamics 与 video prediction。

这一类模型共享最彻底，但推理若保留 dense video tokens 与多步 denoising，仍然慢。
因此后续工作开始把“video modeling 的训练收益”与“test-time 显式生成”拆开。

#### 3. Representation-only / action-centered WAM：训练有视频，推理少生成或不生成

这一族是 NAV 当前最该借鉴的主线。

##### UVA（Unified Video Action Model, RSS 2025）

- 论文：https://arxiv.org/abs/2503.00200
- 项目页：https://unified-video-action-model.github.io/

UVA 的核心是 **joint video-action latent representation** 与 **decoupled
video-action decoding**。它通过统一 latent 表示桥接视频与动作，但输出端使用两个轻量
diffusion heads 解耦 video/action decoding；推理时可跳过 video generation 直接做 action
inference。项目页也明确指出：video generation 作为训练监督可提升 policy，而不降低
policy inference speed。

对 NAV 的启发：我们可以让 VideoGen Mode 与 Policy Mode 共享 Register / selected DiT
hidden interface，但使用不同 input adapter 和 output head。

##### Fast-WAM（2026）

- 论文：https://arxiv.org/abs/2603.16666
- 项目页：https://yuantianyuan01.github.io/FastWAM/
- 代码：https://github.com/yuantianyuan01/FastWAM

Fast-WAM 直接提出核心问题：WAM 的收益来自 test-time future imagination，还是来自
training-time video modeling？它保留 video co-training，但在推理时去掉显式 future
prediction，直接从 latent world representation 生成动作。官方项目页报告：

- backbone：Wan2.2-5B video DiT + 1B action expert；
- 训练：joint action prediction + video modeling；
- 推理：只保留 current observation clean latent tokens，video backbone 单次处理后
  直接 action generation；
- 延迟：单 RTX 5090D V2 32GB 上约 190ms；
- 关键消融：去掉 video co-training 比去掉 test-time imagination 造成更大性能下降。

对 NAV 的启发：Stage Three 不应每步生成完整未来视频；video generation 更适合作为
Stage One/Two 的 representation-shaping supervision，然后 policy 读取 compact Register
和小型 action-query tokens。

##### GigaWorld-Policy / GigaWorld-Policy-0.5（2026）

- 论文：https://arxiv.org/abs/2603.17240
- 项目页：https://gigaai-research.github.io/GigaWorld-Policy/
- 代码：https://github.com/open-gigaai/giga-world-policy

GigaWorld-Policy 指出 Joint WAM 的两个瓶颈：future visual dynamics 与 action joint
reasoning 推理昂贵；visual/motion 表示纠缠使 action accuracy 依赖 future video quality。
它转向 **action-centered WAM**：

```text
current observation -> predict future action sequence
predicted action + same observation -> generate future video as training supervision
inference -> action-only decoding, optional video generation closed
```

公开 repo 的 GigaWorld-Policy-0.5 进一步采用 mixed AC-WM + WAM pretraining，并引入
Mixture-of-Transformers，把 visual dynamics modeling 与 action generation 分给专门
experts，action-only inference 时减少 active computation；README 报告 RTX 4090 本地
约 85ms latency。

对 NAV 的启发：Policy Mode 应以 action/query 为中心；VideoGen branch 可以作为训练时
约束与可选分析输出，而不是每次导航决策的必经路径。

##### ImageWAM（2026）

- 论文：https://arxiv.org/abs/2606.19531
- 项目页：https://zhangwenyao1.github.io/ImageWAM/
- 代码：https://github.com/yuyangalin/ImageWAM

ImageWAM 进一步质疑 video generation 本身是否必要。它指出 video-based WAM 有三类
开销/风险：dense multi-frame future tokens 昂贵；full video prediction 会消耗容量于
action-irrelevant appearance details；长程 imagination 错误会误导 action。它改用 image
editing foundation model，建模 source-grounded target-frame transformation，使表示更关注
action-relevant visual changes。

对 NAV 的启发：即便我们保留 video generation 训练，policy 端也应主动压缩/筛选
action-relevant tokens，避免把容量浪费在导航无关的像素细节上。

##### Faster-WAM / Faster-WAM-family（2026）

- Faster-WAM efficient future conditioning：https://arxiv.org/abs/2608.04404
- Faster-WAM deep action module critique：https://arxiv.org/abs/2608.02365

这组最新工作针对 Fast-WAM 后的 trade-off：完全移除 test-time future representation
虽然快，但可能损失 OOD generalization；保留 joint future-action interaction 又慢。它们
提出 sparse future conditioning、SparseMoT、Interval KV-Fusion 等机制，在少数层/少数
阶段做 video-action interaction，避免每层深融合。

对 NAV 的启发：Stage Three 可考虑“稀疏读取 selected 3D/Video hidden layers”，而不是
每层都让 policy/action tokens 与 dense video tokens 深度交互。

### 复现目标选择

#### 目标一：Fast-WAM

| 项 | 内容 |
| --- | --- |
| 选择理由 | 最直接验证“video co-training 训练有用、test-time future generation 可去掉” |
| 代码成熟度 | 官方 repo，README 包含 environment、model preparation、dataset、inference、training |
| 复现 benchmark | 优先 LIBERO smoke；条件允许再 RoboTwin |
| 关键复现项 | released checkpoint inference latency；video co-training vs w/o video co-training 消融若权重可得 |
| 与 NAV 对应 | 对应 NAV Stage Three 的 compact policy mode；验证不生成未来视频也能保留 WAM 收益 |

建议本地路径：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_fastwam
/sharedata/FastWAM
NAV/result/fastwam/
NAV/log/fastwam/
```

第一阶段只做：

```text
clone -> env -> download released checkpoint / minimal benchmark assets
-> single-task inference smoke -> latency logging
```

#### 目标二：GigaWorld-Policy

| 项 | 内容 |
| --- | --- |
| 选择理由 | action-centered WAM，与 NAV 的“policy 小输入 + video branch 训练约束”最接近 |
| 代码成熟度 | 官方 repo 已公开，安装入口清晰；包含 GigaWorld-Policy-0.5 描述 |
| 复现 benchmark | 优先官方最小推理或 RoboTwin smoke；再看是否能复现 85ms latency |
| 关键复现项 | action-only inference latency；Mixture-of-Transformers / expert routing 是否真的绕开 video branch |
| 与 NAV 对应 | 对应 NAV 的 mode-specific adapter + shared hidden interface 设计 |

建议本地路径：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_gigaworld_policy
/sharedata/GigaWorld-Policy
NAV/result/gigaworld_policy/
NAV/log/gigaworld_policy/
```

第一阶段只做：

```text
clone -> env -> inspect model config / checkpoint availability
-> minimal inference or model-forward smoke
-> confirm action-only path vs optional video path
```

### 第二梯队

| 工作 | 暂不作为第一批复现的原因 | 后续用途 |
| --- | --- | --- |
| UVA | 思想非常干净，但 2025 RSS、与当前 Wan/DreamZero 系路线距离更远 | 写 related work；若 Fast-WAM/GigaWorld 复现受阻，转为候选 |
| ImageWAM | 更像“用 image editing 替代 video generation”的反命题，不是严格 video-gen co-training | 作为 action-relevant visual transformation 对照 |
| Faster-WAM | 2026-08 很新，代码/权重状态需进一步确认 | 后续用于设计 sparse layer fusion |

### 对 NAV Backbone/Input 设计的归纳

近期工作给出的共同答案是：

```text
不要让 policy 和 video generation 强行共用同一种 dense video input。
应该共用 latent/world representation、memory interface 与 selected backbone layers，
但使用 mode-specific input adapters 和 output heads。
```

对应 NAV：

```text
VideoGen Mode:
  dense noisy future latent
  + local memory
  + Register
  + text/action/timestep condition
  -> shared DiT / video branch
  -> video flow/noise head

Policy Mode:
  compact current obs tokens
  + Register
  + action/query tokens
  + instruction/goal condition
  -> selected shared DiT layers or policy adapter
  -> action / waypoint / subgoal head
```

训练时保留 video generation 与 3D supervision；推理时优先 policy-only，不显式生成
dense future video。Register 是二者的共同 memory interface，selected DiT hidden layer
是二者共享的 geometry/world representation interface。

### 2026 主会 / Workshop 算法趋势补充（2026-08-11）

本节只记录算法趋势，不替代 `NAV/essay/relatedworks/related_works_survey.md`
中的论文写作表述。写论文时必须继续区分 main conference、workshop、arXiv
technical report 和第三方列表待核条目。

#### 趋势一：WAM 的争论焦点从“能否生成未来”转向“未来建模怎样服务动作”

2026 的 WAM / robot world-model 论文里，最接近 NAV 的共同问题是：

```text
future visual dynamics 是训练信号，还是推理时必经中间变量？
```

GigaWorld-Policy、Fast-WAM、mimic-video、World Action Models are Zero-shot
Policies 和 RSS 2026 的 Interactive World Simulator 给出的答案并不完全相同：

- GigaWorld / Fast-WAM：训练时保留 future visual dynamics，推理时 action-only；
- mimic-video：冻结视频骨干，在潜空间规划，再用轻量 IDM / action decoder 转动作；
- Interactive World Simulator：把 action-conditioned video world model 作为数据引擎
  与评测代理，而非直接把 video branch 嵌入 policy；
- World Action Models are Zero-shot Policies：强调 video/world-action 表征本身可
  直接产生可执行动作，但当前证据多在机器人操作域。

对 NAV 的直接约束：Stage Three 不应把 dense future video generation 作为导航闭环
必经路径；future video / 3D consequence 更适合作为 Stage One/Two 的
representation-shaping supervision。

#### 趋势二：长程 interactive video world model 竞争点集中在 memory、cache 与几何一致性

ICLR/ICML/CVPR 2026 相关工作（Astra、Vid2World、LIVE、WorldPlay、GenieDrive、
VGGT-Ω、RELIC 等）把竞争点集中在：

- 自回归 / autoregressive denoising 如何减小 train-test gap；
- memory / KV cache / compressed history 如何覆盖 long horizon；
- 几何或 4D occupancy 如何约束视频世界模型；
- 实时交互时如何兼顾 responsiveness 与 coherence。

对 NAV 的直接约束：NAV 的“长程”不应只用 target video horizon 变长来证明，而应
明确拆成：

```text
short consequence prediction:
  T_latent=4 / 13 RGB frames

long memory span:
  Register rollout over IW-equivalent 1/4/8/16 chunks
```

这使 NAV 的核心假设变成“predict short, remember long”，与纯长视频生成路线区分。

#### 趋势三：VLN / navigation 正从端到端 VLA 走向显式中间结构

RSS 2026 的 OpenFrontier 和 SuperMap 代表另一条与 NAV 接近但不同的路线：

- OpenFrontier：把导航表述为 sparse subgoal identification and reaching，用
  visual-language grounded frontiers 作 semantic anchors，强调 training-free 和系统简单；
- SuperMap：用可查询 4D scene graph 作为 VLN / embodied AI 的统一空间记忆；
- AdaNav、DualVLN、FantasyVLN 等 2026 OpenReview/CVPR 线索则强调 MLLM reasoning、
  uncertainty-adaptive reasoning 或 fast-slow dual system。

对 NAV 的直接约束：related work 不能只说“现有 VLN 缺少 memory”。更准确的说法是：

```text
现有 VLN 正在引入显式 frontier、map、scene graph、uncertainty reasoning 和
fast-slow systems；NAV 的不同在于把 memory 来源从导航任务定制结构，换成由
action-conditioned world modeling 与 geometry supervision 训练出的 online Register。
```

#### 趋势四：会议论文写法更强调单一科学命题，tech report 更强调系统路线图

VGGT-Ω（CVPR 2026 Oral）是最值得 NAV 学的会议论文写法：它围绕
“feed-forward reconstruction 如何 scale”组织 Method，把 architecture、loss、
dynamic representation 和 self-supervised training 都放在同一命题下。

GigaWorld-Policy、GigaWorld-Policy-0.5、Qwen-RobotWorld、Infinite-World 等更像
tech report / model report：可以展开讲数据、pretraining pipeline、系统速度与 gallery。
NAV 若按会议论文写，应避免 tech-report 式堆模块，而应围绕一个核心命题组织：

```text
How can a streaming world-model memory become a navigation memory?
```

### 相关资源

- DreamZero 本地复现文档：`NAV/doc/01_model/model_evolution_and_current_architecture.md`
- BridgeVLA++ memory 复现：`NAV/doc/01_model/model_evolution_and_current_architecture.md`
- NAV Stage One：`NAV/doc/01_model/model_evolution_and_current_architecture.md`
- NAV Stage Two：`NAV/doc/01_model/model_evolution_and_current_architecture.md`
- NAV Stage Three：`NAV/doc/01_model/model_evolution_and_current_architecture.md`
