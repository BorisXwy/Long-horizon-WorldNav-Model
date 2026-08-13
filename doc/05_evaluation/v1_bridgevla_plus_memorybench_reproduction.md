# V1 BridgeVLA++ memoryBench 复现记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-003` |
| 类型 | 外部基线复现 |
| 状态 | Smoke Reproduced |
| 更新时间 | 2026-08-08 |
| 职责 | 记录 BridgeVLA++ memory-enhanced 版本在 memoryBench 上的本地复现命令、结果和限制 |

## 当前结论

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

## 结果路径

| 结果 | 路径 | 说明 |
| --- | --- | --- |
| 3-step 链路 smoke | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/bridgevla_plus_memorybench/smoke/model_160/seed608/result.jsonl` | 4/4 rollout 正常结束，但 `MAX_STEPS=3`，全部未完成任务 |
| 25-step 有效 smoke | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/result/bridgevla_plus_memorybench/smoke_25step/model_160/seed608/result.jsonl` | 4/4 success，实际每条 `nsteps=12` |

25-step result 摘要：

| task filter | taskvars 数 | episodes/taskvar | max steps | success |
| --- | ---: | ---: | ---: | ---: |
| `put_block_back` | 4 | 1 | 25 | 4 / 4 |

## 复现命令

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

## 运行观测

- server 首次加载慢，主要在 PaliGemma 与 `model_160.pth` 初始化；二次加载因系统 cache
  明显更快。
- 25-step smoke 中每个 taskvar 用时约 12 秒；server 端每个动作 step 对应一次
  `POST /predict`。
- GPU0 在 server 加载后显存约 9.7GB；client 主要负责仿真 rollout。
- client 退出时出现 `QMutex: destroying locked mutex`，但进程返回码为 0，结果文件正常写入。

## 与官方完整指标的关系

官方 BridgeVLA++ 页面与 README 报告 memoryBench 上 BridgeVLA++ 约 99.7 success rate，
原 BridgeVLA 约 11.3。本文档的 4-episode smoke 不能替代官方完整评测；如需正式复现，
应补齐并展开全部 memoryBench test task，使用官方默认：

- all taskvars；
- `NUM_EPISODES=25`；
- `MAX_STEPS=25`；
- 固定 seed；
- 输出到 `NAV/result/bridgevla_plus_memorybench/full_eval/<run_id>/`。

## 已知限制

1. 本次为了快速确认 memory-enhanced 版本可用，只展开并评测了 `test/put_block_back`。
2. `train/put_block_back` 曾在官方解压脚本执行中被中断后保留了部分已展开目录；zip 原包保留，
   后续训练或完整审计前应重新运行官方解压脚本补齐。
3. `hf-mirror` 对 `hqfang/memorybench` 的 metadata 请求不兼容；缺失 zip 最终使用
   Hugging Face resolve 直链 `wget -c` 下载完成。
