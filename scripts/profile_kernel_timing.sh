#!/bin/bash

# Kernel Timing Profiling Script
# This focuses specifically on kernel execution times
# Usage: ./profile_kernel_timing.sh <your_program> [args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    echo "Example: $0 ./build/zseeker2 megaman"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "Profiling kernel execution timing..."
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""

# Run with nvprof focusing on kernel execution
# --print-gpu-summary: Shows all kernels with their execution times
# --normalized-time-unit ms: Show times in milliseconds
nvprof --device-buffer-size 512MB \
       --print-gpu-summary \
       --normalized-time-unit ms \
       --log-file nvprof_kernel_timing.txt \
       "$PROGRAM" $ARGS

echo ""
echo "Kernel timing profile complete!"
echo "Results saved to: nvprof_kernel_timing.txt"
echo ""
echo "Looking for kernel execution times..."
grep -i "kernel\|compareToZeta" nvprof_kernel_timing.txt || echo "No kernel execution found in trace (kernel may not be completing)"

