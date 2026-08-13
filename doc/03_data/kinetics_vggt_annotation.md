# Kinetics-400 相机视角变化抽样

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-004` |
| 类型 | 标注协议与记录（Annotation Protocol） |
| 状态 | Running |
| 更新时间 | 2026-07-28 |
| 职责 | 定义 VGGT 相机运动标注、RE10K 标定和恢复方式 |

## 设置

- 数据根目录：`/sharedata/datasets/kinetics400`
- 抽样数量：100 条视频
- 抽样方式：对 train、validation、test 三个划分分层随机抽样
- 每条视频：均匀抽取 8 帧
- 模型：VGGT-1B，本地权重
- 输出目录：`NAV/result/kinetics_vggt_camera_100`

尺度无关指标包括：

- 累计相机旋转角（Cumulative Camera Rotation）；
- VGGT 相机路径长度除以中位预测场景深度
  （Camera Path / Median Scene Depth）。

本次启发式分级为：

- 低变化（Low）：累计旋转小于 2°，且位移/深度小于 0.02；
- 中等变化（Moderate）：累计旋转小于 8°，且位移/深度小于 0.10；
- 明显变化（Strong）：其余样本。

## 结果

| 等级 | 数量 | 比例 |
| --- | ---: | ---: |
| 低变化 | 10 | 10% |
| 中等变化 | 15 | 15% |
| 明显变化 | 75 | 75% |

- VGGT 推理成功：100/100；
- 累计旋转角中位数：24.55°；
- 相机路径/场景深度中位数：0.238。

## 解释与限制

结果说明 Kinetics-400 中存在大量跨 chunk 的明显视角变化，适合用于训练
Register/history memory 的长程变化建模。但 Kinetics 包含大量快速人体运动、
剪辑和非刚体物体；VGGT 可能把动态主体造成的对应关系变化解释为相机运动。
因此这些指标适合作为预筛选信号，不应直接视为相机位姿真值。正式构造训练集时，
还应结合 VGGT confidence、镜头切换检测和光流一致性过滤。
## Kinetics-400 全量 VGGT 相机运动标注

### 标注目标

对 `/sharedata/datasets/kinetics400` 的 train、val、test 全部 episode 均匀抽取
8 帧，用 VGGT-1B 预测相机外参和稠密深度，并为每条视频保存：

- 首尾旋转角、累计旋转角；
- 相机轨迹长度、深度归一化平移量；
- `low / moderate / strong` 相机运动标签；
- VGGT 稠密深度置信度的 p10、p25、p50、p75；
- `low / reference / high` 置信度等级及 `motion_label_reliable` 标记。

结果位于 `NAV/result/kinetics_vggt_camera_all/metrics/`，运行日志位于
`NAV/result/kinetics_vggt_camera_all/logs/`。JSONL 每完成一个 episode 就立即
落盘，可中断后使用同一命令续跑。

### RE10K 置信度标定

标定输入是 `/sharedata/RealEstate10K/vggt_runs` 中已有的 30 个场景。VGGT 的
confidence 由 `1 + exp(logit)` 得到，是**稠密深度置信度**，不是相机位姿正确
概率。其绝对值和这批 RE10K 的 Sim(3) 对齐相机中心误差没有显著相关性，因此
仅用它对运动标签做质量门控。

门槛保存在 `NAV/config/vggt_re10k_confidence.json`：

- `reference`：episode 的 p25 ≥ 1.9536 且 p50 ≥ 5.9850，即不低于 RE10K
  场景分布的下四分位；
- `high`：p25 ≥ 6.7623 且 p50 ≥ 12.7652，即达到 RE10K 场景中位水平；
- 其余为 `low`，仍保存运动数值和标签，但令 `motion_label_reliable=false`。

VGGT 官方可视化同样使用 confidence percentile 过滤低置信点。Kinetics 含大量
人物快速运动、剪辑和弱几何场景，校准 smoke test 的 3/3 条均低于 RE10K 门槛；
这不是程序失败，而是明确标出其相对 RE10K 的域外低置信状态。

### 运行与恢复

单 GPU 全量运行：

```bash
bash NAV/scripts/run_kinetics_vggt_all.sh 0 1 cuda:0
```

脚本按视频路径稳定哈希分片。将来若有多张空闲 GPU，可令第二个参数为总分片数，
分别启动不同的 shard index。每个 shard 都读取已有 JSONL 并跳过已完成视频。
