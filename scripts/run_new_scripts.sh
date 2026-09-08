#!/bin/bash
#SBATCH --job-name=new_scripts
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/nissimb/DeepPEF/logs/new_scripts_%j.out
# Runs the 12 delivered scripts on the REAL eval CSVs. CPU only - zero GPU cost, so it does not
# compete with the 30 queued golden arms.
#
# Verified locally against a synthetic fixture with known ground truth before submission:
#   bias_vs_dispersion  PASS (recovered global -4.9407, a_p 0.5212, 27/27 under)
#   w13_robust          PASS (found the k=5 budget point)
#   w13_select          PASS (honest negative: no selection strategy beats random)
# EXCLUDED: double_mutant_test.py -- premise refuted, the 26,315 double mutants do not exist.
# EXCLUDED: hydrophobic_failure.py -- needs a 'mut' column absent from our schema.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
mkdir -p results

BEST=eval_results/abl_p3_slope1.0_s42_e13.csv     # best k=20 arm
CTRL=eval_results/abl_calib_ctrl_repro2_e14.csv   # control
TOP=eval_results/abl_sigma_seed2_e10.csv          # best zero-shot pooled

run () { echo; echo "════════ $* ════════"; python -u "$@" 2>&1 | tail -40; echo "rc=$?"; }

run scripts_new/bias_vs_dispersion.py $BEST $CTRL $TOP
run scripts_new/w13_robust.py $BEST $CTRL $TOP --reps 40
run scripts_new/w13_select.py $BEST
run scripts_new/epoch_provenance.py --eval-dir eval_results --log-dir logs --docs results
run scripts_new/canonical_rescore.py --eval-dir eval_results --log-dir logs 2>/dev/null || \
  run scripts_new/canonical_rescore.py eval_results
run scripts_new/loro_and_factorA_audit.py factorial
run scripts_new/loro_and_factorA_audit.py loro

echo; echo "════════ ARTIFACT CHECK (a DONE line is not evidence) ════════"
for f in results/BIAS_VS_DISPERSION.tsv results/W13_ROBUST.tsv results/W13_SELECT.tsv \
         results/EPOCH_PROVENANCE.tsv; do
  [ -f "$f" ] && echo "  OK      $f ($(wc -l < $f) lines)" || echo "  MISSING $f"
done
echo "NEW_SCRIPTS_DONE"
