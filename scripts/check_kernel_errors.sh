#!/bin/bash

# Kernel Error Check Script using compute-sanitizer
# Usage: ./check_kernel_errors.sh <your_program> [args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    echo "Example: $0 ./build/zseeker2 megaman"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "Checking for kernel errors with compute-sanitizer..."
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""

# Check if compute-sanitizer is available
if ! command -v compute-sanitizer &> /dev/null; then
    echo "Error: compute-sanitizer not found. It should be in your CUDA toolkit."
    echo "Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Run with compute-sanitizer to detect memory errors and other issues
compute-sanitizer --tool=memcheck \
                   --print-limit=100 \
                   --log-file=compute_sanitizer.log \
                   "$PROGRAM" $ARGS

echo ""
echo "Error check complete!"
echo "Results saved to: compute_sanitizer.log"

