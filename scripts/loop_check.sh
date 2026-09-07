#!/bin/bash
# THE LOOP CHECK. Run this every round instead of asking "is there anything to do?".
# It answers three questions in one shot: what finished that I have not read, what is
# still running, and what is next in TASKS.md.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
echo "=== 1. WORK READY TO COLLECT (act on these first) ==="
for d in Megascale-fineTuning/models/*kf_gld_* Megascale-fineTuning/models/*kf_loro*; do
  [ -d "$d" ] || continue
  T=$(basename "$d"|sed 's/.*light_attentionkf_//')
  e=$(ls "$d"/kf_all_epoch_*.pt 2>/dev/null|sed 's/.*epoch_//;s/\.pt//'|sort -n|tail -1)
  [ -z "$e" ] && continue
  if [ "$e" -ge 14 ] && ! ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then
    echo "  READY-TO-SCORE  $T (epoch $e)"
  fi
done
n=0
for d in Megascale-fineTuning/models/*kf_p3_*; do
  [ -f "$d/kf_all_epoch_14.pt" ] || continue
  T=$(basename "$d"|sed 's/.*light_attentionkf_//')
  ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1 || n=$((n+1))
done
echo "  unscored finished factorial cells: $n"
echo "  factorial CSVs: $(ls eval_results/abl_p3_a*.csv 2>/dev/null|wc -l)/48"
echo
echo "=== 2. STILL RUNNING ==="
printf "  public %s/5   golden %s/8   cpu %s   pending %s\n" \
  "$(squeue -u nissimb -h -t RUNNING -o '%P'|grep -c '^gpu$')" \
  "$(squeue -u nissimb -h -t RUNNING -o '%P'|grep -c rtx6000)" \
  "$(squeue -u nissimb -h -t RUNNING -o '%P'|grep -c '^cpu$')" \
  "$(squeue -u nissimb -h -t PENDING|wc -l)"
for d in Megascale-fineTuning/models/*kf_gld_* Megascale-fineTuning/models/*kf_loro*; do
  [ -d "$d" ] || continue
  T=$(basename "$d"|sed 's/.*light_attentionkf_//')
  e=$(ls "$d"/kf_all_epoch_*.pt 2>/dev/null|sed 's/.*epoch_//;s/\.pt//'|sort -n|tail -1)
  printf "  %-22s epoch %s/14\n" "$T" "${e:-0}"
done
echo
echo "=== 3. NEXT OPEN TASK ==="
grep -m3 '^- \[ \]' results/TASKS.md 2>/dev/null || echo "  TASKS.md has no open items"
echo
echo "=== 4. HEALTH ==="
# The autopilot runs as a SLURM job on a COMPUTE node, so pgrep on a login node finds
# nothing and reports a false death. Ask SLURM instead.
if squeue -u nissimb -h -o "%j %T" | grep -q "^autopilot RUNNING"; then
  echo "  autopilot ALIVE (SLURM job $(squeue -u nissimb -h -o '%i %j' | awk '$2=="autopilot"{print $1}'))"
else
  echo "  autopilot DEAD <-- resubmit: sbatch --partition=cpu --qos normal --time 7-00:00:00 --wrap 'bash /home/nissimb/auto/run_autopilot.sh'"
fi
tail -1 ~/auto/autopilot.log 2>/dev/null | sed 's/^/  /'
