#!/usr/bin/env bash
# Netie AI — Container Registry Push Script

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
PROJECT="${PROJECT:-netie-ai}"
IMAGE_NAME="edge-qc-inference"
TAG="${1:-latest}"

FULL_IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo " Netie AI — Registry Push"
echo " Image : ${FULL_IMAGE}"
echo "=============================================="

echo "[1/2] Building Docker image..."
docker build -t "${FULL_IMAGE}" -f deployments/Dockerfile.tensorrt .

echo "[2/2] Pushing to Registry..."
docker push "${FULL_IMAGE}"

echo "✅ Successfully pushed: ${FULL_IMAGE}"
