#!/bin/bash

# Kernel Memory Leak and Issue Detection Script using compute-sanitizer
# Usage: ./check_kernel_leaks.sh [--quick|--full] <program> [args...]
#
# This script runs multiple compute-sanitizer tools to detect:
# - Memory leaks
# - Memory errors (out-of-bounds, use-after-free, etc.)
# - Race conditions
# - Uninitialized memory reads
# - Synchronization errors
#
# Options:
#   --quick  : Only run memcheck (memory leaks and errors) - faster
#   --full   : Run all checks (default)

MODE="full"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--quick|--full] <program> [args...]"
    echo "Example: $0 --quick ./build/zseeker2 megaman"
    echo "Example: $0 --full ./build/zseeker2 megaman"
    exit 1
fi

PROGRAM="$1"
shift
ARGS="$@"

echo "=========================================="
echo "Kernel Memory Leak and Issue Detection"
echo "=========================================="
echo "Mode: $MODE"
echo "Program: $PROGRAM"
echo "Args: $ARGS"
echo ""

# Check if compute-sanitizer is available
if ! command -v compute-sanitizer &> /dev/null; then
    echo "Error: compute-sanitizer not found. It should be in your CUDA toolkit."
    echo "Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Create logs directory
LOG_DIR="compute_sanitizer_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Logs will be saved to: $LOG_DIR/"
echo ""

# Function to run a sanitizer check
run_check() {
    local tool=$1
    local description=$2
    local log_file="$LOG_DIR/${tool}_${TIMESTAMP}.log"
    
    echo "----------------------------------------"
    echo "Running: $description"
    echo "Tool: $tool"
    echo "Log: $log_file"
    echo "----------------------------------------"
    
    # Add leak-check options for memcheck
    local extra_opts=""
    if [ "$tool" = "memcheck" ]; then
        extra_opts="--leak-check=full --show-leak-kinds=all"
    fi
    
    compute-sanitizer --tool=$tool \
                       --print-limit=100 \
                       --log-file="$log_file" \
                       --error-exitcode=0 \
                       $extra_opts \
                       "$PROGRAM" $ARGS
    
    local exit_code=$?
    echo ""
    
    # Analyze the log for key issues
    if [ -f "$log_file" ]; then
        local error_count=$(grep -c "ERROR SUMMARY" "$log_file" || echo "0")
        local leak_count=$(grep -c "LEAK SUMMARY" "$log_file" || echo "0")
        
        if [ "$tool" = "memcheck" ]; then
            local leaks_found=$(grep -A 5 "LEAK SUMMARY" "$log_file" | grep -E "definitely lost|indirectly lost|possibly lost" | grep -v "0 bytes" | wc -l || echo "0")
            if [ "$leaks_found" -gt 0 ]; then
                echo "⚠️  MEMORY LEAKS DETECTED!"
                grep -A 10 "LEAK SUMMARY" "$log_file" | head -15
            else
                echo "✓ No memory leaks detected"
            fi
        fi
        
        echo "Check complete (exit code: $exit_code)"
    fi
    echo ""
    
    return $exit_code
}

# Run checks based on mode
echo "Starting kernel analysis (mode: $MODE)..."
echo ""

# 1. Memory Check (includes leak detection) - always run
run_check "memcheck" "Memory Errors and Leaks"

if [ "$MODE" = "full" ]; then
    # 2. Race Condition Check
    run_check "racecheck" "Race Conditions"
    
    # 3. Uninitialized Memory Check
    run_check "initcheck" "Uninitialized Memory Reads"
    
    # 4. Synchronization Check
    run_check "synccheck" "Synchronization Errors"
fi

# Generate summary
echo "=========================================="
echo "Summary Report"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "All logs saved to: $LOG_DIR/"
echo ""
echo "Individual reports:"

TOOLS="memcheck"
if [ "$MODE" = "full" ]; then
    TOOLS="memcheck racecheck initcheck synccheck"
fi

for tool in $TOOLS; do
    log_file="$LOG_DIR/${tool}_${TIMESTAMP}.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "--- $tool ---"
        
        # Extract key metrics from each log
        if [ "$tool" = "memcheck" ]; then
            if grep -q "LEAK SUMMARY" "$log_file"; then
                echo "Leak Summary:"
                grep -A 5 "LEAK SUMMARY" "$log_file" | head -6
            fi
            if grep -q "ERROR SUMMARY" "$log_file"; then
                error_line=$(grep "ERROR SUMMARY" "$log_file" | tail -1)
                echo "$error_line"
            fi
        else
            if grep -q "ERROR SUMMARY" "$log_file"; then
                error_line=$(grep "ERROR SUMMARY" "$log_file" | tail -1)
                echo "$error_line"
            fi
        fi
    fi
done

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "For detailed information, check the individual log files in: $LOG_DIR/"
echo ""
echo "Tips:"
echo "  - Memory leaks will show in the memcheck log"
echo "  - Check for 'definitely lost' or 'indirectly lost' bytes"
echo "  - Use '--leak-check=full' for more detailed leak info (slower)"
echo ""

