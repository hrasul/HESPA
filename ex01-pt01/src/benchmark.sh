#!/bin/bash

set -e

# Configuration
WORKSPACE_DIR="/home/cip/2025/fo35cece/HESP/ex01-pt01"
BUILD_DIR="${WORKSPACE_DIR}/build/stream"
SRC_DIR="${WORKSPACE_DIR}/src/stream"
RESULTS_FILE="${WORKSPACE_DIR}/bandwidth_measurements.txt"

# Benchmark parameters
WARM_UP_ITERATIONS=2
MEASUREMENT_ITERATIONS=10
NUM_TRIALS=3

# Buffer sizes (in number of doubles, each double is 8 bytes)
# L1 cache (~32 KB)
# L2 cache (~256 KB) 
# L3 cache (~8 MB)
# Main memory (64+ MB)
BUFFER_SIZES=(
    "4096"           # 32 KB - L1
    "32768"          # 256 KB - L2
    "131072"         # 1 MB - L3 partial
    "1048576"        # 8 MB - L3
    "8388608"        # 64 MB - Main memory
    "16777216"       # 128 MB - Main memory
    "33554432"       # 256 MB - Main memory
)

cd "${SRC_DIR}"
make clean > /dev/null 2>&1 || true
make all > /dev/null 2>&1

cat > "${RESULTS_FILE}" << 'EOF'

STREAM Benchmark Results: Sustained CPU Bandwidth

Machine: Casa Huber CIP reference machine
Buffer Size analysis: L1, L2, L3, and Main Memory

Buffer Size (MB) | Version    | Trial | MLUP/s     | Bandwidth (GB/s)
================================================================================
EOF

for buffer_size in "${BUFFER_SIZES[@]}"; do
    buffer_mb=$((buffer_size * 8 / 1024 / 1024))
    
    # Serial version
    echo -n "Testing buffer size ${buffer_mb} MB (serial)... "
    for trial in $(seq 1 ${NUM_TRIALS}); do
        output=$("${BUILD_DIR}/stream-base" "${buffer_size}" "${WARM_UP_ITERATIONS}" "${MEASUREMENT_ITERATIONS}" 2>&1)
        mlups=$(echo "$output" | grep "MLUP/s:" | awk '{print $2}')
        bandwidth=$(echo "$output" | grep "bandwidth:" | awk '{print $2}')
        
        printf "%6d | %-10s | %5d | %10.2f | %15s\n" "${buffer_mb}" "serial" "${trial}" "${mlups}" "${bandwidth}" >> "${RESULTS_FILE}"
    done
    echo "done"
    
    # Parallel version
    echo -n "Testing buffer size ${buffer_mb} MB (parallel)... "
    for trial in $(seq 1 ${NUM_TRIALS}); do
        output=$("${BUILD_DIR}/stream-omp-host" "${buffer_size}" "${WARM_UP_ITERATIONS}" "${MEASUREMENT_ITERATIONS}" 2>&1)
        mlups=$(echo "$output" | grep "MLUP/s:" | awk '{print $2}')
        bandwidth=$(echo "$output" | grep "bandwidth:" | awk '{print $2}')
        
        printf "%6d | %-10s | %5d | %10.2f | %15s\n" "${buffer_mb}" "parallel" "${trial}" "${mlups}" "${bandwidth}" >> "${RESULTS_FILE}"
    done
    echo "done"
done


cat >> "${RESULTS_FILE}" << 'EOF'
Summary Statistics (Average Bandwidth per Configuration)

Buffer Size (MB) | Serial Avg (GB/s) | Parallel Avg (GB/s) | Speedup
================================================================================
EOF

for buffer_size in "${BUFFER_SIZES[@]}"; do
    buffer_mb=$((buffer_size * 8 / 1024 / 1024))
    
    serial_bw=$(grep "^[[:space:]]*${buffer_mb} | serial" "${RESULTS_FILE}" | awk '{print $NF}' | sed 's/ GB\/s//' | awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
    parallel_bw=$(grep "^[[:space:]]*${buffer_mb} | parallel" "${RESULTS_FILE}" | awk '{print $NF}' | sed 's/ GB\/s//' | awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
    
    if [[ "$serial_bw" != "N/A" && "$parallel_bw" != "N/A" ]]; then
        speedup=$(echo "scale=2; $parallel_bw / $serial_bw" | bc)
        printf "%6d | %17s | %19s | %8s\n" "${buffer_mb}" "${serial_bw} GB/s" "${parallel_bw} GB/s" "${speedup}x" >> "${RESULTS_FILE}"
    fi
done

echo " Results saved to: ${RESULTS_FILE}"
