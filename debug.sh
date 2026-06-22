#!/bin/bash
# Kengine Debug Test Script
# Run from the project root (Kengine/) or adjust DEMO_PATH below

set -euo pipefail

# Configuration
DEMO_PATH="${1:-./home/riverdk/projects/git/Vulkan/Kengine/kengine_demo}"  # Default to debug build; pass path as argument
LOG_DIR="debug_logs"
VALGRIND_LOG="${LOG_DIR}/valgrind_$(date +%Y%m%d_%H%M%S).log"
SUPPRESSIONS="valgrind_suppressions.supp"

mkdir -p "$LOG_DIR"

echo "=== Kengine Debug Test Script ==="
echo "Demo path: $DEMO_PATH"
echo "Logs will be saved to $LOG_DIR/"
echo ""

usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  run          - Normal run"
    echo "  valgrind     - Run with Valgrind (Memcheck)"
    echo "  gdb          - Run under GDB"
    echo "  vg-gdb       - Valgrind + GDB (for advanced debugging)"
    echo "  all          - Run all tests sequentially"
    echo "  help         - Show this help"
    exit 1
}

if [[ $# -eq 0 || "$1" == "help" ]]; then
    usage
fi

case "$1" in
    run)
        echo "Running normal execution..."
        "$DEMO_PATH"
        ;;
    
    valgrind)
        echo "Running under Valgrind..."
        valgrind --tool=memcheck \
                 --leak-check=full \
                 --show-leak-kinds=all \
                 --track-origins=yes \
                 --num-callers=30 \
                 ${SUPPRESSIONS:+"--suppressions=$SUPPRESSIONS"} \
                 --log-file="$VALGRIND_LOG" \
                 "$DEMO_PATH"
        echo "Valgrind log saved to: $VALGRIND_LOG"
        echo "Tip: Check for real leaks vs. 'still reachable' from Vulkan/GLFW/VMA."
        ;;
    
    gdb)
        echo "Launching GDB..."
        echo "Inside GDB, use: run, break main, bt (backtrace), etc."
        gdb --args "$DEMO_PATH"
        ;;
    
    vg-gdb)
        echo "Running Valgrind with GDB support..."
        echo "In another terminal run: gdb ./build-debug/kengine_demo then 'target remote | vgdb'"
        valgrind --vgdb-error=0 --log-file="${LOG_DIR}/vgdb.log" "$DEMO_PATH"
        ;;
    
    all)
        echo "Running all tests..."
        "$0" run
        "$0" valgrind
        echo "GDB test skipped in 'all' mode (interactive). Run manually if needed."
        ;;
    
    *)
        usage
        ;;
esac

echo ""
echo "Done."
