#!/bin/bash
#SBATCH --job-name=fp_ddg
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/home/nissimb/DeepPEF/logs/fp_ddg_%j.out
# CONFIRMED: FireProtDB holds real mutations for our proteins (1W4H_A:L167A, H142W, ...).
# The "exactly 2 per protein" pattern was an artefact of head-truncation, not the data.
#
# THE DECISIVE QUESTION: do those rows carry a MEASURED ddG value? A mutation identifier alone is
# useless for W13 -- we need the number. This extracts full rows for our proteins and reports
# which columns hold numeric ddG.
#
# WHY IT MATTERS: W13 lifts pooled 0.5845 -> 0.6993 across all 52 runs but needs k measured
# mutations per protein, which today implies a wet-lab campaign. If FireProtDB supplies them,
# the method becomes a lookup. That is the largest practical change available to this thesis.
#
# Read-only stream from shaharax's tree. Nothing is written there.
cd /home/nissimb/DeepPEF
Z=data/FireProtDB/raw/fireprotdb_dump.zip
P="2K5H|6EWT|1W4H|6EWS|1GYZ|1QP2|1TUC|1PSE|1QKH|2KWH|2K1B|2KXD|2KVS|2LQK|2PTL|2RU9|6OBK|2L9R|6M3N|2KYB|1I6C"
OUT=/home/nissimb/fp_rows.tsv
echo "=== extracting FULL rows for our proteins (this streams 4.8 GB) ==="
unzip -p "$Z" 01_fireprotdb_2025-09-20.sql 2>/dev/null | grep -iE "($P)" > "$OUT"
echo "rows captured: $(wc -l < "$OUT")"
echo ""
echo "=== how many rows carry a plausible ddG (a signed decimal)? ==="
grep -cE "$(printf '%s' '-?[0-9]+\.[0-9]+')" "$OUT"
echo ""
echo "=== 12 sample rows, full width ==="
head -12 "$OUT" | cut -c1-400
echo ""
echo "=== per-protein row counts ==="
grep -ioE "($P)" "$OUT" | tr 'a-z' 'A-Z' | sort | uniq -c | sort -rn
echo "FP_DDG_DONE"
