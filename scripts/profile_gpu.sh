#!/bin/bash

# GPU Profiling Script
# This script monitors GPU utilization while your program runs

echo "Starting GPU monitoring..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping GPU monitoring..."
    kill $MONITOR_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start monitoring GPU utilization in the background
(
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
        sleep 1
    done
) &
MONITOR_PID=$!

# Wait for the monitoring process
wait $MONITOR_PID

