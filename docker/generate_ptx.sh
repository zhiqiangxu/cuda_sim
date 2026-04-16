#!/bin/bash
# Generate PTX from .cu files using Docker (no GPU needed)
#
# Usage:
#   ./docker/generate_ptx.sh kernel.cu              # outputs kernel.ptx
#   ./docker/generate_ptx.sh kernel.cu -o out.ptx   # custom output path
#   ./docker/generate_ptx.sh src/*.cu                # batch mode

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="cuda-sim-ptxgen"

# Build Docker image if not exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building Docker image '$IMAGE_NAME' (first time only)..."
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

# Parse arguments
OUTPUT=""
INPUT_FILES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            INPUT_FILES+=("$1")
            shift
            ;;
    esac
done

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <kernel.cu> [-o output.ptx]"
    echo "       $0 <file1.cu> <file2.cu> ...  (batch mode)"
    exit 1
fi

for INPUT in "${INPUT_FILES[@]}"; do
    BASENAME=$(basename "$INPUT" .cu)
    INPUT_ABS=$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")
    INPUT_DIR=$(dirname "$INPUT_ABS")

    if [ -n "$OUTPUT" ]; then
        PTX_OUT="$OUTPUT"
    else
        PTX_OUT="${INPUT_DIR}/${BASENAME}.ptx"
    fi

    echo "Generating PTX: $INPUT → $PTX_OUT"

    docker run --rm \
        -v "$INPUT_DIR:/workspace/src" \
        "$IMAGE_NAME" \
        nvcc -ptx "/workspace/src/$(basename "$INPUT")" \
             -o "/workspace/src/${BASENAME}.ptx"

    if [ -n "$OUTPUT" ] && [ "$PTX_OUT" != "${INPUT_DIR}/${BASENAME}.ptx" ]; then
        mv "${INPUT_DIR}/${BASENAME}.ptx" "$PTX_OUT"
    fi

    echo "Done: $PTX_OUT"
done
