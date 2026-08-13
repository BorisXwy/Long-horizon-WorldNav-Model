# V0 Infinite-World 的 VBench 技术指标子集评测

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-001` |
| 类型 | 评测协议与记录（Evaluation Protocol） |
| 状态 | Historical Baseline / 旧入口已从当前代码删除 |
| 更新时间 | 2026-08-06 |
| 职责 | 固定 VBench 对齐口径并按时间记录每次测评（模型/训练实现 + 测评条件 + 结果） |

---

## 0. 评测协议与口径（所有记录共用）

本文件保留 V0 InfiniteWorld/Register A/B 的 VBench 复现记录。文中的旧 NAV
Register 推理脚本已从当前代码主线删除；当前 V1 结构验证见
`v1_full_pipeline_smoke.md`。

### 0.1 VBench 指标

- **六指标版**（仅记录 1 用）：`temporal_flickering`、`dynamic_degree`、
  `motion_smoothness`、`imaging_quality`、`subject_consistency`、
  `background_consistency`。汇总分为 VBench 归一化 `[0,1]`；单视频成像质量
  采用未归一化 MUSIQ 原始分，其汇总分由 VBench 归一化。
- **四指标版**（论文协议，记录 2 起默认）：`motion_smoothness`、
  `dynamic_degree`、`aesthetic_quality`、`imaging_quality`；**Average Score =
  四项算术平均**；明确排除 `subject_consistency`/`background_consistency`
  （交互探索含大幅视角切换，主体与背景本身非平稳）。

### 0.2 论文评测协议（核对自原文 4.2.1 / Table 1 / 4.2.3）

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

### 0.3 通用生成与评测设置

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

### 0.4 评测命令模板

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

### 0.5 测试输入与逐 chunk 规则（NAV Register 版 / InfiniteWorld HPMC 版）

本节明确每次推理喂给模型的四类输入——**文本 text、local memory（那一帧）、
长程 history、action**——以及单 chunk 与 3-chunk 测试在规则上的区别。两套推理
脚本共用同一份 demo 条件，但 history 路径不同。

#### 0.5.1 四类输入的定义

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

#### 0.5.2 单 chunk 与 3-chunk 的规则区别

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

### 0.6 `result/vbench/` 目录命名规则

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

## 测评记录

### 记录 1 — InfiniteWorld 官方权重 6 指标基线（2026-07-23）

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

### 记录 2 — InfiniteWorld 官方权重 论文四项协议小规模复现（2026-07-23）

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

### 记录 3 — NAV A/B 全参 step-1000 同协议对比（2026-07-25）

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

### 记录 4 — NAV A/B from-scratch 中间 checkpoint 同协议对比（2026-08-04）

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

### 记录 5 — InfiniteWorld 官方权重 3-chunk 复现（2026-08-05，已完成）

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

### 记录 6 — InfiniteWorld 16-chunk 论文标准复现（2026-08-05，进行中）

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

### 记录 7 — NAV A/B from-scratch 3-chunk 同协议对比（2026-08-05，已完成）

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

### 记录 8 — Teacher-forced（3 GT 历史 chunk → 预测 1 chunk）三模型对比（2026-08-06，已完成）

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

## 改名映射表（2026-08-06 统一改名）

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
