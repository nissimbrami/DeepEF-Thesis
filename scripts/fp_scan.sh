#!/bin/bash
#SBATCH --job-name=fp_scan
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/nissimb/DeepPEF/logs/fp_scan_%j.out
# Does FireProtDB contain measured ddG for OUR 28 test proteins? If it does, W13's calibration
# mutations come from published data instead of a new wet-lab campaign -- which turns a few-shot
# method that needs a lab into one that needs a lookup.
# The dump is 4.8 GB inside a zip on shaharax's READ-ONLY tree. Stream it, never extract, never
# write there.
cd /home/nissimb/DeepPEF
Z=data/FireProtDB/raw/fireprotdb_dump.zip
OUT=/home/nissimb/fp_hits.txt
# our 21 PDB-coded test proteins (the other 7 are designed sequences with no PDB entry)
P="2K5H|6EWT|1W4H|6EWS|1GYZ|1QP2|1TUC|1PSE|1QKH|2KWH|2K1B|2KXD|2KVS|2LQK|2PTL|2RU9|6OBK|2L9R|6M3N|2KYB|1I6C"
echo "streaming the dump, matching: $P"
unzip -p "$Z" 01_fireprotdb_2025-09-20.sql 2>/dev/null | grep -oiE "($P)" | sort | uniq -c | sort -rn > "$OUT"
echo "=== PDB codes of ours found in FireProtDB ==="
cat "$OUT"
echo ""
echo "distinct proteins of ours present: $(wc -l < "$OUT")"
echo "FP_SCAN_DONE"
