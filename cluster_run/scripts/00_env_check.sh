#!/bin/bash
# UNTESTED: requires cluster -- no cluster access from the build machine; this only inspects the
# environment. It runs no training. Phase 0 (docs/COMPUTE_PLAN.md): verify env + torch/CUDA and
# find where Shahar's processed tensors live. Run on a login node (or a short interactive alloc)
# BEFORE anything else.
#
# SELF-CONTAINED NOTE: this package (cluster_run/) is handed to the cluster agent on its own, but
# the training entrypoint train.py lives in the surrounding DeepPEF repo. Set REPO_ROOT to the
# checkout of shaharec/DeepPEF (see README.md version marker for the commit). Default assumes this
# folder sits at <repo>/cluster_run/.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
echo "REPO_ROOT=${REPO_ROOT}  (override with: REPO_ROOT=/path/to/DeepPEF bash scripts/00_env_check.sh)"
cd "${REPO_ROOT}"

echo "==== 0. cwd ===="
pwd

echo "==== 1. conda env (expected: esm2_env_py38 per sbatch_wtanchor.sh) ===="
module load anaconda 2>/dev/null || echo "  (module load anaconda failed -- check the module system)"
source activate esm2_env_py38 2>/dev/null && echo "  activated esm2_env_py38" \
  || echo "  TODO(operator): esm2_env_py38 not found -- confirm the env name for this account"

echo "==== 2. torch + CUDA ===="
python - <<'PY'
try:
    import torch
    print("  torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device", torch.cuda.get_device_name(0),
              "vram_GB", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
except Exception as e:
    print("  torch import FAILED:", e)
PY

echo "==== 3. sbatch partitions/QOS visible to this account ===="
sinfo -o "%P %a %l %G" 2>/dev/null | head -20 || echo "  sinfo unavailable"
echo "  TODO(operator): COMPUTE_PLAN prefers cs-4090-*; cs-6000-*/ee-l40s-* for memory-bound arms."

echo "==== DONE: env check. Next: scripts/01_data_check.sh ===="
