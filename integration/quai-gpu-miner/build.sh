#!/bin/bash
# Build quai-gpu-miner with cuda_sim in Docker
# Run from cuda_sim root directory: ./integration/quai-gpu-miner/build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_SIM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Building quai-gpu-miner with cuda_sim ==="
echo "cuda_sim root: $CUDA_SIM_ROOT"
echo ""

cd "$CUDA_SIM_ROOT"

# Build Docker image
docker build \
    -f integration/quai-gpu-miner/Dockerfile \
    -t quai-gpu-miner-cuda-sim \
    . 2>&1

echo ""
echo "=== Build complete ==="
echo "To extract the binary:"
echo "  docker create --name tmp quai-gpu-miner-cuda-sim"
echo "  docker cp tmp:/build/quai-gpu-miner/build/kawpowminer/kawpowminer ."
echo "  docker rm tmp"
