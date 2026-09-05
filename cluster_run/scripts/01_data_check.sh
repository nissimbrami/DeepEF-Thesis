#!/bin/bash
# UNTESTED: requires cluster -- the training data is not on the build machine; this only locates
# tensors and the reference eval CSV. Phase 0 (docs/COMPUTE_PLAN.md): find Shahar's processed
# tensors on THIS cluster BEFORE downloading anything -- calib_ctrl was produced here.
#
# SELF-CONTAINED NOTE: set REPO_ROOT to the shaharec/DeepPEF checkout (see README.md version marker).
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
cd "${REPO_ROOT}"

echo "==== 1. training data (train.py reads the OLD K50 layout) ===="
for d in ./data/Processed_K50_dG_datasets/training_data \
         ./data/Processed_K50_dG_datasets/mutation_datasets \
         /mnt/new_groups/keasar_group/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/training_data; do
  if [ -d "$d" ]; then echo "  FOUND: $d ($(ls "$d" | wc -l) entries)"; else echo "  missing: $d"; fi
done

echo "==== 2. real Shahar per-mutation eval CSV (validates calib_diag to 0.77-0.81) ===="
# This is the deferred end-to-end validation of the primary instrument (docs/STATUS.md). The build
# machine had no such CSV, so calib_diag's affine-oracle could not be confirmed locally.
ER=/mnt/new_groups/keasar_group/casp15/Shahar/DeepPEF/eval_results
if [ -d "$ER" ]; then
  echo "  FOUND eval_results: $ER"; ls "$ER" | head
  echo "  -> then: python cluster_run/code/calib_diag.py --csv <one_eval.csv>   (expect affine-oracle 0.77-0.81)"
else
  echo "  TODO(operator): $ER not visible from here -- locate the per-mutation eval CSV on this cluster."
fi

echo "==== DONE: data check. Next: scripts/02_smoke.sh ===="
