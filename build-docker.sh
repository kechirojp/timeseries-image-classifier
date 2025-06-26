#!/bin/bash

# Docker Hub用のビルドスクリプト
# Usage: ./build-docker.sh [tag_version]

# デフォルトタグ
DEFAULT_TAG="latest"
TAG=${1:-$DEFAULT_TAG}

# Docker Hub ユーザー名（変更してください）
DOCKER_HUB_USER="kechirojp"
IMAGE_NAME="timeseries-image-classifier"

echo "Building Docker image: ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"

# Dockerイメージをビルド
docker build -t ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG} .

# ビルドが成功した場合の確認
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📦 Image: ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To push to Docker Hub, run:"
    echo "docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run the container:"
    echo "docker run --gpus all -it ${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"
else
    echo "❌ Build failed!"
    exit 1
fi
