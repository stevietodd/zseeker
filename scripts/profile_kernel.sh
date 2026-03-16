#!/bin/bash

# Kernel Profiling Script using nsys (NVIDIA Nsight Systems)
# This is a general-purpose profiling script
# Usage: ./profile_kernel.sh <your_program> [args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    echo "Example: $0 ./build/zseeker2 megaman"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "Profiling kernel execution with nsys..."
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys (NVIDIA Nsight Systems) not found!"
    echo "It should be included with your CUDA toolkit."
    echo "Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Profile with nsys to get kernel timing and metrics
nsys profile \
    --output=nsys_profile_${TIMESTAMP} \
    --trace=cuda,nvtx \
    --stats=true \
    --force-overwrite=true \
    "$PROGRAM" $ARGS

echo ""
echo "Profiling complete!"
echo "Results saved to:"
echo "  - nsys_profile_${TIMESTAMP}.nsys-rep (binary profile - can be viewed with nsys-ui)"
echo ""
echo "To view the profile interactively, run:"
echo "  nsys-ui nsys_profile_${TIMESTAMP}.nsys-rep"
echo ""
echo "Or generate text stats:"
echo "  nsys stats --report gputrace --report cudaapis nsys_profile_${TIMESTAMP}.nsys-rep"

