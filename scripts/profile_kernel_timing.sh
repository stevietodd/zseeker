#!/bin/bash

# Kernel Timing Profiling Script using nsys
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

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys (NVIDIA Nsight Systems) not found!"
    echo "It should be included with your CUDA toolkit."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="kernel_timing_${TIMESTAMP}.txt"

# Run with nsys focusing on GPU trace (kernel execution times)
nsys profile \
    --output=nsys_timing_${TIMESTAMP} \
    --trace=cuda \
    --stats=true \
    --force-overwrite=true \
    "$PROGRAM" $ARGS > "$OUTPUT_FILE" 2>&1

echo ""
echo "Kernel timing profile complete!"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Looking for kernel execution times..."
grep -i "compareToZeta5loop\|Duration\|Total\|Time(\%)" "$OUTPUT_FILE" | head -30 || echo "No kernel execution found in trace (kernel may not be completing)"

# Also try to get stats
if nsys stats --report gputrace nsys_timing_${TIMESTAMP}.nsys-rep 2>/dev/null | tee -a "$OUTPUT_FILE"; then
    echo ""
    echo "Detailed stats appended to: $OUTPUT_FILE"
fi