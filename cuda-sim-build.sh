#!/bin/bash
# cuda-sim-build: Complete build pipeline for running CUDA on CPU
#
# Usage:
#   ./cuda-sim-build.sh kernel.cu main.cpp -o app
#
# Pipeline:
#   kernel.cu → [Docker nvcc -ptx] → kernel.ptx → [ptx2cpp.py] → kernel_cpu.cpp → [g++] → app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/tools"
INCLUDE_DIR="$SCRIPT_DIR/include"

# Parse arguments
CU_FILES=()
CPP_FILES=()
OUTPUT="a.out"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -O*)
            EXTRA_FLAGS="$EXTRA_FLAGS $1"
            shift
            ;;
        *.cu)
            CU_FILES+=("$1")
            shift
            ;;
        *.cpp|*.cc)
            CPP_FILES+=("$1")
            shift
            ;;
        *)
            EXTRA_FLAGS="$EXTRA_FLAGS $1"
            shift
            ;;
    esac
done

if [ ${#CU_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <kernel.cu> [main.cpp] [-o output]"
    exit 1
fi

GENERATED_CPP=()
TEMP_FILES=()

cleanup() {
    for f in "${TEMP_FILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

# Step 1 & 2: For each .cu file, generate PTX then translate to C++
for CU in "${CU_FILES[@]}"; do
    BASENAME=$(basename "$CU" .cu)
    PTX_FILE="/tmp/${BASENAME}_$$.ptx"
    CPP_FILE="/tmp/${BASENAME}_cpu_$$.cpp"
    TEMP_FILES+=("$PTX_FILE" "$CPP_FILE")

    echo "[1/3] Generating PTX: $CU"
    "$SCRIPT_DIR/docker/generate_ptx.sh" "$CU" -o "$PTX_FILE"

    echo "[2/3] Translating PTX → C++: $PTX_FILE"
    python3 "$TOOLS_DIR/ptx2cpp.py" "$PTX_FILE" -o "$CPP_FILE"

    GENERATED_CPP+=("$CPP_FILE")
done

# Step 3: Compile everything
echo "[3/3] Compiling → $OUTPUT"
g++ -std=c++17 -I "$INCLUDE_DIR" \
    "${CPP_FILES[@]}" "${GENERATED_CPP[@]}" \
    $EXTRA_FLAGS \
    -o "$OUTPUT"

echo "Build complete: $OUTPUT"
