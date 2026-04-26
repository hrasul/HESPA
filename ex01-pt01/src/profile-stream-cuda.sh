#!/bin/bash

set -e

WORKSPACE_DIR="/home/cip/2025/fo35cece/HESP/ex01-pt01"
BUILD_DIR="${WORKSPACE_DIR}/build"
APP="${BUILD_DIR}/stream/stream-cuda"
OUTDIR="${BUILD_DIR}/profiles"
OUT="${WORKSPACE_DIR}/bandwidth_cuda_results.txt"

NX_LIST=(4096 32768 131072 1048576 8388608 16777216 33554432)
BS_LIST=(64 128 256 512 1024)

WARMUP=2
NIT=10
TRIALS=3


cat > "$OUT" << EOF

CUDA STREAM Bandwidth Results

nx         | blockSize | trial | MLUP/s       | Bandwidth (GB/s)
--------------------------------------------------------------
EOF

for nx in "${NX_LIST[@]}"; do
  for bs in "${BS_LIST[@]}"; do
    for trial in $(seq 1 "$TRIALS"); do
      output=$($APP "$nx" "$WARMUP" "$NIT" "$bs")
      mlups=$(echo "$output" | grep "MLUP/s:" | awk '{print $2}')
      bw=$(echo "$output" | grep "bandwidth:" | awk '{print $2}')

      printf "%-10s | %-9s | %-5s | %-12s | %-16s\n" \
        "$nx" "$bs" "$trial" "$mlups" "$bw" >> "$OUT"
    done
  done
done

cat >> "$OUT" << EOF

Average Bandwidth

nx         | blockSize | Avg Bandwidth (GB/s)
--------------------------------------------------------------
EOF

for nx in "${NX_LIST[@]}"; do
  for bs in "${BS_LIST[@]}"; do
    avg_bw=$(awk -F'|' -v nx="$nx" -v bs="$bs" '
      $1 ~ /^[[:space:]]*[0-9]/ {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $5);
        if ($1 == nx && $2 == bs) {
          sum += $5; count++;
        }
      }
      END {
        if (count > 0) printf "%.4f", sum / count;
      }
    ' "$OUT")

    printf "%-10s | %-9s | %-20s\n" "$nx" "$bs" "$avg_bw" >> "$OUT"
  done
done