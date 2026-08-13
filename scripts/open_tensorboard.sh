#!/usr/bin/env bash
set -euo pipefail

# 启动只包含指定 NAV 实验的 TensorBoard。
# 用法：
#   bash scripts/open_tensorboard.sh <实验名>[,<实验名>...] [端口]
#
# 示例：
#   bash scripts/open_tensorboard.sh re10k-all-a-full-ebs4-from-infinite
#   bash scripts/open_tensorboard.sh \
#     re10k-all-a-full-ebs4-from-infinite,re10k-all-b-full-ebs4-from-infinite

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "用法: bash scripts/open_tensorboard.sh <实验名>[,<实验名>...] [端口]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAV_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world"
LOG_ROOT="${NAV_ROOT}/log"
SESSION_NAME="nav_tensorboard"
EXPERIMENTS="$1"
PORT="${2:-6007}"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "找不到 InfiniteWorld Python 环境: ${ENV_ROOT}" >&2
    exit 1
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "无效端口: ${PORT}" >&2
    exit 2
fi

IFS=',' read -r -a RUN_NAMES <<< "${EXPERIMENTS}"
if (( ${#RUN_NAMES[@]} == 0 )); then
    echo "至少需要提供一个实验名。" >&2
    exit 2
fi

LOGDIR_SPEC=""
for run_name in "${RUN_NAMES[@]}"; do
    # 只允许直接位于 NAV/log 下的实验目录，避免路径穿越或误加载其他目录。
    if [[ -z "${run_name}" || "${run_name}" == */* || "${run_name}" == "." || "${run_name}" == ".." ]]; then
        echo "无效实验名: ${run_name}" >&2
        exit 2
    fi

    tensorboard_dir="${LOG_ROOT}/${run_name}/tensorboard"
    if [[ ! -d "${tensorboard_dir}" ]]; then
        echo "找不到实验的 TensorBoard 日志: ${tensorboard_dir}" >&2
        echo "现有实验：" >&2
        find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type d -name tensorboard \
            -printf '  %h\n' 2>/dev/null | sed "s#${LOG_ROOT}/##" | sort >&2
        exit 1
    fi

    if [[ -n "${LOGDIR_SPEC}" ]]; then
        LOGDIR_SPEC+=","
    fi
    LOGDIR_SPEC+="${run_name}:${tensorboard_dir}"
done

# 该环境是打包后的 Conda 环境，没有标准 bin/activate；显式设置与激活等价的
# 关键变量，并在 tmux 中使用绝对 Python 路径，避免登录 shell 改写环境。
export CONDA_PREFIX="${ENV_ROOT}"
export PATH="${ENV_ROOT}/bin:${PATH}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" \
    "${ENV_ROOT}/bin/python -m tensorboard.main \
    --logdir_spec='${LOGDIR_SPEC}' \
    --host 0.0.0.0 \
    --port '${PORT}' \
    --reload_interval 5"

for _ in $(seq 1 20); do
    if NO_PROXY=127.0.0.1,localhost \
        curl -fsS "http://127.0.0.1:${PORT}/data/runs" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! NO_PROXY=127.0.0.1,localhost \
    curl -fsS "http://127.0.0.1:${PORT}/data/runs" >/dev/null 2>&1; then
    echo "TensorBoard 启动失败，tmux 输出如下：" >&2
    tmux capture-pane -pt "${SESSION_NAME}" -S -80 >&2 || true
    exit 1
fi

SERVER_HOST="$(hostname)"
SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
VISIBLE_RUNS="$(
    NO_PROXY=127.0.0.1,localhost \
        curl -fsS "http://127.0.0.1:${PORT}/data/runs"
)"

echo "TensorBoard 已启动，仅加载以下实验："
printf '  %s\n' "${RUN_NAMES[@]}"
echo "TensorBoard API 可见 runs: ${VISIBLE_RUNS}"
echo
if [[ -n "${SERVER_IP}" ]]; then
    echo "服务器网络直连: http://${SERVER_IP}:${PORT}"
fi
echo "本地端口转发（在本地终端运行）:"
echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} xiewenyuan@${SERVER_HOST}"
echo "本地浏览器地址: http://localhost:${PORT}"
echo
echo "查看服务日志:"
echo "  tmux attach -t ${SESSION_NAME}"

# 在 VS Code Remote 终端中，端口会由 .vscode/settings.json 自动转发；
# remote-cli 会要求本地 VS Code 打开已经转发的 localhost 地址。
if [[ -n "${VSCODE_IPC_HOOK_CLI:-}" ]] && command -v code >/dev/null 2>&1; then
    sleep 1
    if code --open-url "http://localhost:${PORT}" >/dev/null 2>&1; then
        echo "已请求 VS Code 打开本地转发页面。"
    else
        echo "VS Code 页面未能自动打开，请在 Ports 面板打开端口 ${PORT}。" >&2
    fi
fi
