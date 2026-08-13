# NAV

NAV 是 `3d_wm_vln` 项目中的 navigation/world-model 方向工作区，包含当前
V1 Register + action-centered world model / VLN 方案的代码、脚本、配置、文档和论文草稿。

## 仓库内容

- `doc/`：项目设计、数据、训练、评测、调研和运行状态文档。
- `src/`：NAV Python 模块与模型组件。
- `scripts/`：数据准备、训练、推理、评测和复现脚本。
- `config/`：可公开的 YAML/JSON 配置。
- `essay/`：论文草稿和写作材料。

## 不进入仓库的内容

以下内容由 `.gitignore` 排除：

- `log/`、`result/` 等运行产物；
- checkpoint / model weights / latent cache；
- 生成视频、TensorBoard、数据缓存；
- 本机环境文件、rclone 配置、token 或密钥。

资源路径和数据/权重位置请查看 `doc/06_operations/resource_inventory.md` 与
`doc/repointro.md`。

## 当前设计入口

- 当前状态：`doc/00_overview/project_status.md`
- 当前硬规则：`doc/00_overview/project_invariants.md`
- 决策日志：`doc/00_overview/decision_log.md`
- V1 模型接口：`doc/01_design/v1_action_centered_io_interface.md`
- V1 三阶段训练计划：`doc/04_training/v1_three_stage_training_data_plan.md`

