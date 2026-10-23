#!/usr/bin/env bash
set -euo pipefail
# A compiler installed in conda needs its matching C++ runtime before torch
# loads a system libstdc++. Keep this scoped to the validation process tree.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "$(id -u)" == 0 ]]; then
  echo "请使用非 root 用户运行验证。" >&2
  exit 1
fi
python -c 'import os; assert os.environ.get("CONDA_DEFAULT_ENV") in ("pice", "pcie"), "请先激活 pcie（或 pice）环境"'
cd "$(dirname "$0")/.."
rank_count="$(python -c 'import os; print(os.environ.get("NPROC_PER_NODE", "2"))')"
run_dir="$(mktemp -d /tmp/pcie-comm-validation.XXXXXX)"
export PCIE_COMM_JIT_CACHE_DIR="$run_dir/jit"
export PCIE_COMM_AUTOTUNE_DIR="$run_dir/autotune"
export PCIE_COMM_AUTOTUNE_TIMER=cuda_event
export MAX_JOBS="$(python -c 'import os; print(os.environ.get("MAX_JOBS", "2"))')"
python -m pytest -q
python -m torch.distributed.run --standalone --nproc_per_node="$rank_count" \
  examples/validate_gpu.py --phase cold --results "$run_dir/results"
python -m torch.distributed.run --standalone --nproc_per_node="$rank_count" \
  examples/validate_gpu.py --phase warm --results "$run_dir/results"
echo "验证报告和缓存目录：$run_dir"
