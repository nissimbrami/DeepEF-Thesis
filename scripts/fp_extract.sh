#!/bin/bash
#SBATCH --job-name=fp_extract
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/nissimb/DeepPEF/logs/fp_extract_%j.out
# The scan found 22 of our proteins with exactly 2 hits each -- suspiciously uniform. Two hits
# per protein across 22 proteins looks like a schema artefact (e.g. a CREATE TABLE comment or an
# index listing), not 22 proteins' worth of measurements. Extract the CONTEXT of each match to
# find out whether these are real ddG rows or noise.
# Read-only stream from shaharax's tree; nothing is written there.
cd /home/nissimb/DeepPEF
Z=data/FireProtDB/raw/fireprotdb_dump.zip
P="2K5H|6EWT|1W4H|6EWS|1GYZ|1QP2|1TUC|1PSE|1QKH|2KWH|2K1B|2KXD|2KVS|2LQK|2PTL|2RU9|6OBK|2L9R|6M3N|2KYB|1I6C"
echo "=== context around each match (first 40) ==="
unzip -p "$Z" 01_fireprotdb_2025-09-20.sql 2>/dev/null \
  | grep -inE "($P)" | head -40 | cut -c1-220
echo ""
echo "=== which SQL tables mention ddG at all? ==="
unzip -p "$Z" 01_fireprotdb_2025-09-20.sql 2>/dev/null \
  | grep -ioE "CREATE TABLE [\`\"a-z_]+" | sort -u | head -30
echo "FP_EXTRACT_DONE"
