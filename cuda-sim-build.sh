#!/bin/bash
# cuda-sim-build: Complete build pipeline for running CUDA on CPU
#
# Usage:
#   ./cuda-sim-build.sh kernel.cu main.cpp -o app
#
# Pipeline:
#   kernel.cu → [Docker nvcc -ptx] → kernel.ptx → [ptx2cpp.py] → kernel_cpu.cpp/.h → [g++] → app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/tools"
INCLUDE_DIR="$SCRIPT_DIR/include"
DOCKER_IMAGE="nvidia/cuda:12.6.0-devel-ubuntu22.04"

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
    echo ""
    echo "Example:"
    echo "  $0 examples/vector_add/kernel.cu examples/vector_add/main.cpp -o vector_add"
    exit 1
fi

GENERATED_CPP=()
GENERATED_HDR_DIRS=()
TEMP_FILES=()

cleanup() {
    for f in "${TEMP_FILES[@]}"; do
        rm -rf "$f"
    done
}
trap cleanup EXIT

# Step 1 & 2: For each .cu file, generate PTX then translate to C++
for CU in "${CU_FILES[@]}"; do
    CU_ABS=$(cd "$(dirname "$CU")" && pwd)/$(basename "$CU")
    CU_DIR=$(dirname "$CU_ABS")
    BASENAME=$(basename "$CU" .cu)
    PTX_FILE="/tmp/${BASENAME}_$$.ptx"
    TMPDIR="/tmp/cuda_sim_$$"
    mkdir -p "$TMPDIR"
    CPP_FILE="$TMPDIR/${BASENAME}_cpu.cpp"
    HDR_FILE="$TMPDIR/kernel_cpu.h"
    TEMP_FILES+=("$PTX_FILE" "$CPP_FILE" "$HDR_FILE" "$TMPDIR")

    echo "[1/3] Generating PTX: $CU → $PTX_FILE"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
    docker run --rm \
        -v "$CU_DIR:/workspace/src" \
        "$DOCKER_IMAGE" \
        nvcc -ptx "/workspace/src/$(basename "$CU")" \
             -o "/workspace/src/__tmp_${BASENAME}.ptx" > /dev/null 2>&1
    mv "$CU_DIR/__tmp_${BASENAME}.ptx" "$PTX_FILE"

    echo "[2/3] Translating PTX → C++: $PTX_FILE"
    python3 "$TOOLS_DIR/ptx2cpp.py" "$PTX_FILE" -o "$CPP_FILE" -H "$HDR_FILE" 2>&1

    GENERATED_CPP+=("$CPP_FILE")
    GENERATED_HDR_DIRS+=("$TMPDIR")
done

# Step 3: Compile everything
echo "[3/3] Compiling → $OUTPUT"
# Build include flags for generated headers
HDR_INCLUDES=""
for d in "${GENERATED_HDR_DIRS[@]}"; do
    HDR_INCLUDES="$HDR_INCLUDES -I$d"
done

g++ -std=c++17 \
    -I "$INCLUDE_DIR/compat" \
    -I "$INCLUDE_DIR" \
    $HDR_INCLUDES \
    "${CPP_FILES[@]}" "${GENERATED_CPP[@]}" \
    $EXTRA_FLAGS \
    -o "$OUTPUT"

echo ""
echo "Build complete: $OUTPUT"
echo "Run with: ./$OUTPUT"
