#!/bin/bash

# Kernel Diagnostic Script
# This script helps determine if your kernel is running or hanging
# Usage: ./diagnose_kernel.sh <your_program> [args...]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    echo "Example: $0 ./build/zseeker2 megaman"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "=== Kernel Diagnostic Tool ==="
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""
echo "This will run your program and monitor GPU activity."
echo "Press Ctrl+C to stop monitoring (program will continue)."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitoring..."
    kill $MONITOR_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start monitoring GPU in the background
(
    echo "Time(s)  GPU%  Mem%  Temp(C)  Power(W)"
    echo "----------------------------------------"
    while true; do
        TIMESTAMP=$(date +%s)
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -v ts="$TIMESTAMP" '{printf "%-8s %-4s %-4s %-7s %-7s\n", ts, $1"%", $2"%", $3"°C", $4"W"}'
        sleep 1
    done
) &
MONITOR_PID=$!

# Run the program
echo "Starting program..."
"$PROGRAM" $ARGS
EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "Program exited with code: $EXIT_CODE"

