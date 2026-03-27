#!/bin/bash
echo "=== Starting MLP experiment ==="
python3 train-2_5_light_att.py --debug --emb_projection mlp 2>&1 | \
  grep -E "(EPOCH|loss=.*lossd=.*lossg=|rank_Ejf|gap\(|Finished|Debug|Build)" | \
  grep -v "Epoch " | grep -v "batch" | tee /tmp/exp_mlp_clean.log

echo ""
echo "=== Starting LOW_RANK experiment ==="
python3 train-2_5_light_att.py --debug --emb_projection low_rank 2>&1 | \
  grep -E "(EPOCH|loss=.*lossd=.*lossg=|rank_Ejf|gap\(|Finished|Debug|Build)" | \
  grep -v "Epoch " | grep -v "batch" | tee /tmp/exp_low_rank_clean.log

echo ""
echo "=== ALL EXPERIMENTS DONE ==="
