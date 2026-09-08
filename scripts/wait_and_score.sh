#!/bin/bash
# Wait for kf_all_epoch_<E>.pt to appear and be fully written, then score it. Args: <tag> <epoch>
T=$1; E=$2
D="/home/nissimb/DeepPEF/Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${T}"
CKPT="$D/kf_all_epoch_${E}.pt"
echo "waiting for $CKPT ..."
for i in $(seq 1 240); do   # up to 2h
  if [ -f "$CKPT" ]; then
    s1=$(stat -c %s "$CKPT"); sleep 20; s2=$(stat -c %s "$CKPT")
    # size stable => write finished (checkpoints are ~264MB)
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 200000000 ]; then echo "checkpoint stable at $s2 bytes"; break; fi
  fi
  sleep 30
done
[ -f "$CKPT" ] || { echo "TIMEOUT: $CKPT never appeared"; exit 1; }
bash /home/nissimb/DeepPEF/scripts/score_one.sh "$T" "$E"
