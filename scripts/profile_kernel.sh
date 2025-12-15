#!/bin/bash

# Kernel Profiling Script using nvprof
# Usage: ./profile_kernel.sh <your_program> [args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    echo "Example: $0 ./build/zseeker2"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "Profiling kernel execution with nvprof..."
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""

# Run with nvprof to get kernel timing and metrics
# --device-buffer-size: Increase buffer to avoid "insufficient device buffer space" warnings
# --print-gpu-trace: Show detailed trace of GPU activities (including kernel execution times)
# --print-api-trace: Show API call trace
# Note: Cannot use both --print-gpu-summary and --print-gpu-trace together
nvprof --device-buffer-size 512MB \
       --print-gpu-trace \
       --print-api-trace \
       --log-file nvprof_trace.txt \
       --export-profile nvprof_profile.nvprof \
       --normalized-time-unit ms \
       "$PROGRAM" $ARGS

echo ""
echo "Profiling complete!"
echo "Results saved to:"
echo "  - nvprof_trace.txt (text trace)"
echo "  - nvprof_profile.nvprof (binary profile - can be viewed with nvvp)"
echo ""
echo "To view the profile interactively, run:"
echo "  nvvp nvprof_profile.nvprof"

