# Minimal image with nvcc for generating PTX — no GPU required
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
