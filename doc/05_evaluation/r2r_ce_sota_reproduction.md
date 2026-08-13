# R2R-CE / RxR-CE SOTA 复现协议（≥ StreamVLN）

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-002` |
| 类型 | 评测协议与复现规范（Evaluation Protocol） |
| 状态 | StreamVLN 100/500-ep Done · DualVLN 100-ep Done · Full val_unseen Pending |
| 更新时间 | 2026-08-08 16:40（Asia/Shanghai） |
| 职责 | 规定 NAV 对 R2R-CE/RxR-CE 上分数不低于 StreamVLN 的可开源复现工作的唯一环境、数据、命令与结果落盘口径；并记录已完成子集复现的成功率与 latency |

## 当前结论

1. **StreamVLN / DualVLN 环境与官方评测入口已跑通**；权重与数据软链就绪。
2. **子集复现已完成**（均为 R2R-CE `val_unseen` **数据集顺序前 N 条**，**非全量 1839**）：
   - StreamVLN：**100-ep**、**500-ep**
   - DualVLN：**100-ep**（`diffusers==0.31.0`，本地 GPU 整模推理）
3. DualVLN 前 100 已贴近论文全量量级；StreamVLN 前 500 仍低于论文全量约 12 SR 点——子集偏差 + 未跑全量，不宜直接判装坏。
4. **推理均在本体（本地 GPU）**：`from_pretrained` 加载整模，无远端 API。DualVLN 的 7B（S2）与小 DiT（S1）同进程同卡。
5. 环境形态与复现门槛见下文；全量 Val-Unseen 仍未跑。

本轮纳入复现队列：

| 优先级 | 方法 | 传感器 | 文献 R2R Val-Unseen SR/SPL | 仓库 | 权重 | 本地环境 |
| --- | --- | --- | ---: | --- | --- | --- |
| P0 | StreamVLN | RGB | **56.4 / 50.2** | `3d_wm_vln/StreamVLN` @`e48f6ff` | `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | `virtual_env/.venv_streamvln` |
| P1 | InternVLA-N1 DualVLN | RGB | **64.3 / 58.5** | `3d_wm_vln/InternNav` @`7a5c624` | `InternRobotics/InternVLA-N1-DualVLN` | `virtual_env/.venv_internnav` |
| P2 | InternVLA-N1 Dual + NavDP* | RGB-D | **64.1 / 58.1**（或旧表 58.2/54.0） | 同上 | `InternRobotics/InternVLA-N1-w-NavDP` | 同上 |

明确不复现、只引用：Qwen-RobotNav、Robostral Navigate、以及 panoramic-only 行（见 `../07_research/r2r_rxr_benchmark_sota.md`）。

工程约束：第三方仓在 `3d_wm_vln/<RepoName>`；环境在 `virtual_env/.venv_<name>`（conda prefix）；日志 `NAV/log/vln_ce/`；结果 `NAV/result/vln_ce/<method>/`。

---

## 复现成功率（Reproduced by us）

口径：Habitat VLN-CE，R2R `val_unseen`，成功距离 3 m；动作物理 **0.25 m / 15°**（以各仓库 yaml/代码为准）。来源标记：`Reproduced by us (partial)`。

### StreamVLN

| Run | N | SR↑ | SPL↑ | OS↑ | NE↓ | 产物 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke100 | 100 | **41.0** | **36.0** | **54.0** | **6.53** | `result/vln_ce/streamvln/smoke100_20260807_220243/` |
| smoke500 | 500 | **44.2** | **37.4** | **58.0** | **6.07** | `result/vln_ce/streamvln/smoke500_20260808_021908/` |
| Paper v1_3（全量） | 1839 | 56.4 | 50.2 | 63.6 | 4.90 | Official reported |

- 官方入口：`StreamVLN/streamvln/streamvln_eval.py`；封装 `MAX_EPISODES=N bash NAV/scripts/eval_baselines/run_streamvln_eval.sh`。
- 采样：数据集顺序前 N；100-ep 约覆盖 7 个 scene。
- 对照 JSON：各 run 下 `metrics/compare_paper.json`。
- 栈：Python 3.9.23 / torch 2.1.2+cu121 / habitat_sim 0.2.4 / `flash_attn==2.5.8`。

### DualVLN（InternVLA-N1 Dual System）

| Run | N | SR↑ | SPL↑ | OS↑ | NE↓ | 产物 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke100 | 100 | **67.0** | **57.4** | **73.0** | **4.33** | `result/vln_ce/dualvln/smoke100_20260808_082720/` |
| Paper（全量） | 1839 | 64.3 | 58.5 | 70.7 | 4.05 | Official reported |

- 官方入口：`InternNav/scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py`；封装 `run_dualvln_eval.sh`。
- 权重：`/sharedata/NAV/baselines/checkpoints/InternVLA-N1-DualVLN`（`system1=nextdit_async`）。
- **关键依赖**：`diffusers==0.31.0`（InternNav#322；新版 Lumina FFN 维与 ckpt 不匹配）。
- 栈：Python 3.9.23 / torch 2.5.1+cu121 / habitat_sim 0.2.4。
- 子集 SR 略高于论文全量属正常波动，**不能**直接宣称超过 SOTA。

### 并排摘要

| 方法 | 子集 | Ours SR/SPL | Paper SR/SPL | ΔSR |
| --- | ---: | ---: | ---: | ---: |
| StreamVLN | 500 | 44.2 / 37.4 | 56.4 / 50.2 | −12.2 |
| DualVLN | 100 | 67.0 / 57.4 | 64.3 / 58.5 | +2.7 |

---

## Latency 体系与实测

机器可读明细：各 run 的 `metrics/latency.json`。下列为日志时间戳解析（无 CUDA event），含环境步进开销。

### StreamVLN：单 Video-LLM 半闭环

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

### DualVLN：S2（7B VLM）+ S1（小 DiT）

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

### 对比（同机子集复现）

| | StreamVLN | DualVLN |
| --- | ---: | ---: |
| 架构 | 单 VLM | S2 7B + S1 ~91M |
| 步进墙钟 | ~2.1 Hz | ~1.05 Hz |
| 大模型调用 | ~0.53 Hz | S2 ~0.27 Hz |
| 快路径 | 窗口内 KV 续写 | S1 DiT ~0.35 s/次 |
| 典型 ep 墙钟 | ~46 s | ~85 s |

---

## 方法或依据

### 门槛与分组规则

- 主比较轴：**R2R-CE Val-Unseen SR**；RxR-CE 同步报告 SR/SPL/nDTW。
- 同一表格内必须标注传感器（RGB / RGB-D / panoramic），禁止跨传感器宣称“超过”。
- 来源列必填：`Official reported` / `Reproduced by us` /
  `Re-evaluated from released checkpoint`（见 NAV-RES-002）。
- 子集结果必须标注 N 与采样方式；不得与全量 Official 数字混称为已复现 SOTA。

### 路径约定

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

### Habitat / 二进制依赖说明

`habitat-sim==0.2.4` 的 conda 构建仅支持 **Python 3.9**，且不能注入纯
`python -m venv` 目录。因此 `.venv_*` 使用：

```bash
conda create -y -p virtual_env/.venv_streamvln python=3.9 pip
conda install -y -p virtual_env/.venv_streamvln habitat-sim=0.2.4 withbullet headless \
  -c conda-forge -c aihabitat
```

其余 PyTorch / 项目依赖用该前缀内的 `pip` 安装。`setup_*.sh` 已按此实现。
InternNav 侧须钉死 **`diffusers==0.31.0`**（见 `setup_internnav_env.sh`）。

### 数据链接（只读软链，不复制）

由 `scripts/eval_baselines/link_vln_ce_data.sh` 创建：

```text
StreamVLN/data/datasets/r2r      → /sharedata/datasets/R2R/R2R_VLNCE_v1-3
StreamVLN/data/datasets/rxr      → /sharedata/datasets/RxR/raw/rxr_ce/RxR_VLNCE_v0
StreamVLN/data/scene_datasets/mp3d → /sharedata/datasets/mp3d/v1/tasks/mp3d
InternNav 侧同样链到上述公共目录（脚本内按官方期望布局调整）
```

InternNav Habitat 场景根应指向 `.../tasks`（使 `mp3d_ce/<scene>` 解析正确），避免双重 `mp3d/`。

### 标准评测命令

```bash
bash NAV/scripts/eval_baselines/link_vln_ce_data.sh
bash NAV/scripts/eval_baselines/setup_streamvln_env.sh
bash NAV/scripts/eval_baselines/setup_internnav_env.sh
bash NAV/scripts/eval_baselines/download_checkpoints.sh

MAX_EPISODES=100 CUDA_VISIBLE_DEVICES=0 bash NAV/scripts/eval_baselines/run_streamvln_eval.sh
MAX_EPISODES=500 CUDA_VISIBLE_DEVICES=1 bash NAV/scripts/eval_baselines/run_streamvln_eval.sh
MAX_EPISODES=100 CUDA_VISIBLE_DEVICES=1 bash NAV/scripts/eval_baselines/run_dualvln_eval.sh
```

### 每次 run 必须记录的字段

写入 `NAV/result/vln_ce/<method>/<run_id>/metrics/`：

- `summary.json` / `compare_paper.json` /（若测过）`latency.json`
- git commit（第三方仓库 + NAV）；venv 与版本；split、N、采样方式
- 传感器、动作步长（本复现为 **0.25 m / 15°**）；checkpoint
- SR / SPL / NE / OS；墙钟；GPU；来源标记 `Reproduced by us`

## 执行状态快照（2026-08-08）

| 步骤 | 状态 | 说明 |
| --- | --- | --- |
| clone / 数据软链 / 权重 | 完成 | StreamVLN + DualVLN ckpt |
| `.venv_streamvln` / `.venv_internnav` | 完成 | 见上文栈 |
| StreamVLN 1-ep / 100-ep / 500-ep | **完成** | 见成功率表 |
| DualVLN 100-ep | **完成** | `diffusers==0.31.0`；GPU1 |
| StreamVLN / DualVLN 全量 val_unseen | 未跑 | `MAX_EPISODES=-1` |
| DualVLN + NavDP* | 未跑 | P2 |
| RxR-CE | 未跑 | — |

## 限制与未决问题

1. `.venv_*` 实际是 conda prefix（非纯 venv），因 habitat-sim 0.2.4 仅 py3.9 conda 包。
2. 全量 Val-Unseen 未跑；子集数字不可替代 Official 全量行。
3. Latency 由日志时间戳估算，非 CUDA profiler；含 Habitat / look-down 开销。
4. DualVLN 官方 req 中的 diffusers 版本与可加载版本不一致，以 **0.31.0** 为准。
5. Qwen-RobotNav / Robostral / panoramic 行不进复现队列。

## 相关资源

- Benchmark 数字与引用链：`../07_research/r2r_rxr_benchmark_sota.md`（NAV-RES-002）
- VLN 数据状态：`../03_data/vln_dataset_preparation_status.md`（NAV-DAT-007）
- 资源总账：`../06_operations/resource_inventory.md`
- 决策：`../00_overview/decision_log.md` DEC-021
- 配置：`../../config/r2r_ce_sota_reproduction.env`
- 脚本：`../../scripts/eval_baselines/`
- StreamVLN 结果：`../../result/vln_ce/streamvln/smoke{100_20260807_220243,500_20260808_021908}/`
- DualVLN 结果：`../../result/vln_ce/dualvln/smoke100_20260808_082720/`
